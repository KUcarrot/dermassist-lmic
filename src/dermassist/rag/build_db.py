"""
03_build_rag_db.py (v2)
=======================
[공통 - 1주차] 오프라인 RAG 지식베이스 구축 — PDF 기반 확장판

변경사항 (v1 → v2):
  - PDF 파일에서 텍스트 자동 추출 (PyMuPDF)
  - 웹 페이지 크롤링 결과(.txt) 지원
  - 문장 단위 청킹 개선 (한국어·영어 혼합 대응)
  - 시드 문서는 fallback + 기본 커버리지 보장용
  - 메타데이터(출처, 페이지번호) 보존
  - 검색 정확도 자동 테스트 (기대 카테고리 대비)

실행: python3 03_build_rag_db.py

PDF 배치 방법:
  data/rag_knowledge/ 폴더에 PDF 파일을 넣으면 자동으로 처리됩니다.
  예시:
    data/rag_knowledge/WHO_skin_guidelines.pdf
    data/rag_knowledge/BAD_melanoma_guideline.pdf
    data/rag_knowledge/dermnet_articles/   (텍스트 파일 폴더)

추가 의존성:
  pip install PyMuPDF
"""

import sys
import json
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import RAG_DIR, RAG_DB_DIR, RAG_CONFIG, CLASS_NAMES


# ============================================================
# 1. 데이터 클래스 정의
# ============================================================
@dataclass
class DocumentChunk:
    """단일 청크 메타데이터."""
    doc_id: str           # 문서 식별자
    title: str            # 문서 제목
    source: str           # 출처 (파일명, URL 등)
    source_type: str      # "pdf" | "txt" | "web" | "seed"
    category: str         # 질환 카테고리
    page_number: int      # PDF 페이지 번호 (해당 시)
    chunk_index: int      # 문서 내 청크 순번
    content: str          # 청크 텍스트


# ============================================================
# 2. PDF 텍스트 추출
# ============================================================
def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """
    PDF에서 페이지별 텍스트 추출.
    PyMuPDF(fitz)를 사용 — 레이아웃 보존, 테이블 텍스트 추출 가능.

    Returns:
        [{"page": 1, "text": "..."}, ...]
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[오류] PyMuPDF 미설치. 설치: pip install PyMuPDF")
        return []

    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = clean_extracted_text(text)
            if len(text.strip()) > 50:  # 표지·빈 페이지 무시
                pages.append({"page": page_num + 1, "text": text})
        doc.close()
        print(f"  [PDF] {pdf_path.name}: {len(pages)}페이지 추출")
    except Exception as e:
        print(f"  [PDF 오류] {pdf_path.name}: {e}")

    return pages


def clean_extracted_text(text: str) -> str:
    """PDF 추출 텍스트 정제: 과도한 공백, 헤더/푸터, 페이지 번호 제거."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[-–]\s*\d+\s*[-–]', '', text)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


# ============================================================
# 3. 카테고리 자동 분류
# ============================================================
CATEGORY_KEYWORDS = {
    "mel": ["melanoma", "멜라노마", "흑색종"],
    "bcc": ["basal cell", "기저세포암", "bcc", "진주빛"],
    "bkl": ["keratosis", "각화증", "seborrheic", "지루"],
    "akiec": ["actinic", "bowen", "광선각화", "보웬"],
    "nv": ["nevus", "nevi", "모반", "점"],
    "df": ["dermatofibroma", "피부섬유종", "fibroma"],
    "vasc": ["vascular", "hemangioma", "혈관", "혈관종"],
}


def infer_category(text: str, filename: str = "") -> str:
    """파일명 + 텍스트 내용에서 카테고리 추정."""
    combined = (filename + " " + text[:500]).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined:
                return cat
    return "general"


# ============================================================
# 4. 개선된 청킹 (문장 단위, 한국어 대응)
# ============================================================
def split_into_sentences(text: str) -> List[str]:
    """한국어·영어 혼합 텍스트를 문장 단위로 분리."""
    text = text.replace('\n\n', ' [PARA] ')
    text = text.replace('\n', ' ')

    # 한국어 종결어미 + 영어 마침표 기준 분리
    sentences = re.split(
        r'(?<=[.!?다요음됨함])\s+(?=[A-Z가-힣\d\[])',
        text
    )

    result = []
    for sent in sentences:
        parts = sent.split('[PARA]')
        for p in parts:
            p = p.strip()
            if p:
                result.append(p)
    return result


def chunk_documents(
    text: str,
    max_chunk_chars: int = 1000,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    문장 단위 청킹.
    max_chunk_chars: 청크 최대 글자 수 (~200~250 토큰)
    overlap_sentences: 다음 청크에 포함할 이전 문장 수 (문맥 보존)
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_length + sent_len > max_chunk_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                overlap = current_chunk[-overlap_sentences:]
                current_chunk = overlap
                current_length = sum(len(s) for s in overlap)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sent)
        current_length += sent_len

    if current_chunk:
        last_chunk = " ".join(current_chunk)
        if not chunks or last_chunk != chunks[-1]:
            chunks.append(last_chunk)

    return chunks


# ============================================================
# 5. 시드 문서 (PDF 없을 때 fallback + 기본 커버리지)
# ============================================================
SEED_DOCUMENTS = [
    {
        "doc_id": "mel_overview",
        "title": "멜라노마 (Melanoma) 개요",
        "category": "mel",
        "content": """멜라노마는 멜라닌세포에서 발생하는 악성 종양으로, 피부암 중 가장 치명적이다.
조기 발견 시 5년 생존율이 99%에 달하나, 전이 후에는 30% 미만으로 급감한다.
주요 위험인자는 과도한 자외선 노출, 다수의 비정형 모반, 가족력, 면역 억제 상태 등이다.
호발 부위는 남성의 경우 체간(등), 여성의 경우 하지이며, 어느 피부 부위에서든 발생 가능하다.
표재확산흑색종(SSM)이 가장 흔한 아형으로 전체의 약 70%를 차지한다.
결절흑색종(NM)은 빠르게 성장하며 예후가 불량하다.
악성흑색점흑색종(LMM)은 일광 노출 부위에, 말단흑자흑색종(ALM)은 손발바닥에 호발한다.""",
    },
    {
        "doc_id": "abcde_rule",
        "title": "ABCDE 자가검진 규칙",
        "category": "general",
        "content": """ABCDE 규칙은 멜라노마 의심 병변을 식별하는 임상 가이드라인이다.
A (Asymmetry, 비대칭): 병변을 반으로 나누었을 때 양쪽이 대칭이 아닌 경우.
B (Border, 경계): 경계가 불규칙하거나, 들쭉날쭉하거나, 흐릿한 경우.
C (Color, 색상): 단일 색상이 아닌 여러 색상(갈색, 검정, 붉은색, 흰색, 파란색)이 혼재.
D (Diameter, 직경): 직경이 6mm 이상인 경우 (연필 지우개 크기 이상).
E (Evolving, 변화): 크기, 모양, 색상이 시간에 따라 변화하는 경우.
위 기준 중 하나라도 해당되면 피부과 전문의 상담을 권고한다.
ABCDE 중 E(변화)가 가장 중요한 단일 지표로 여겨진다.
추가로 Ugly Duckling Sign(주변과 다른 점)도 보조적 감별 기준으로 활용된다.""",
    },
    {
        "doc_id": "bcc_overview",
        "title": "기저세포암 (Basal Cell Carcinoma) 개요",
        "category": "bcc",
        "content": """기저세포암은 가장 흔한 피부암으로, 전체 피부암의 약 80%를 차지한다.
자외선 노출이 많은 두경부에 호발하며, 전이는 극히 드물다.
임상적으로 진주빛 반투명 결절(pearly translucent papule)과 나뭇가지 모양 혈관(arborizing telangiectasia)이 가장 특징적인 소견이다.
진주빛 광택이 나는 반투명 구진 또는 결절이 기저세포암을 의심하는 핵심 단서이다.
중앙부 궤양과 rolled border(둘레가 말린 경계)도 진단적 단서이다.
아형으로 결절형, 표재형, 침윤형, 미세결절형이 있으며 결절형이 가장 흔하다.
치료는 외과적 절제가 표준이며, 모스 수술이 가장 높은 완치율(99%)을 보인다.""",
    },
    {
        "doc_id": "bkl_overview",
        "title": "양성 각화증 (Benign Keratosis) 개요",
        "category": "bkl",
        "content": """양성 각화증에는 지루각화증, 일광흑색점, 편평태선양각화증이 포함된다.
지루각화증은 가장 흔한 양성 피부 종양으로, 중년 이후 호발한다.
임상적으로 기름진 표면(stuck-on appearance)과 뿔 낭종(horn cysts)이 특징이다.
대부분 치료가 필요 없으나, 멜라노마와의 감별진단이 중요하다.
급격한 변화나 출혈이 동반되면 조직검사를 고려해야 한다.
Leser-Trélat sign: 지루각화증이 갑자기 다수 발생하면 내부 악성종양을 의심해야 한다.""",
    },
    {
        "doc_id": "akiec_overview",
        "title": "광선각화증 / 보웬병 (Actinic Keratosis) 개요",
        "category": "akiec",
        "content": """광선각화증은 만성 자외선 노출에 의한 전암성 병변이다.
치료하지 않으면 약 5-10%가 편평세포암으로 진행할 수 있다.
보웬병은 표피내 편평세포암(SCC in situ)으로, 광선각화증보다 한 단계 진행된 상태이다.
임상적으로 홍반성 인설 반(erythematous scaly patch)이 특징이며,
일광 노출 부위인 얼굴, 두피, 전완부에 호발한다.
촉진 시 사포(sandpaper) 같은 거친 감촉이 특징적이다.
치료 옵션에는 냉동요법, 국소 5-FU, 이미퀴모드, 광역동치료(PDT) 등이 있다.""",
    },
    {
        "doc_id": "nv_overview",
        "title": "멜라닌세포모반 (Melanocytic Nevi) 개요",
        "category": "nv",
        "content": """멜라닌세포모반은 가장 흔한 양성 멜라닌세포 종양으로, 일반적으로 '점'이라고 불린다.
경계형, 복합형, 진피내 모반으로 분류되며, 대부분 후천적으로 발생한다.
비정형(이형성) 모반은 멜라노마 위험인자로, ABCDE 기준에 부합할 수 있다.
50개 이상의 모반을 가진 경우 멜라노마 위험이 4-5배 증가한다.
일반적인 모반은 치료가 필요 없으나, 변화가 관찰되면 전문의 평가가 필요하다.""",
    },
    {
        "doc_id": "df_overview",
        "title": "피부섬유종 (Dermatofibroma) 개요",
        "category": "df",
        "content": """피부섬유종은 진피층의 양성 섬유조직구성 종양이다.
주로 하지에 호발하며, 젊은 성인 여성에서 더 흔하다.
촉진 시 단단한 결절로 만져지며, 측면 압박 시 함몰되는 dimple sign이 특징적이다.
외상이나 곤충 자상에 대한 반응으로 발생할 수 있다.
대부분 무증상이며 치료가 필요 없으나, 미용적 이유로 절제할 수 있다.""",
    },
    {
        "doc_id": "vasc_overview",
        "title": "혈관병변 (Vascular Lesions) 개요",
        "category": "vasc",
        "content": """혈관병변에는 혈관종, 혈관각화종, 화농성 육아종 등이 포함된다.
혈관종은 영아기에 호발하며, 자연 퇴행하는 경우가 많다.
혈관각화종은 중년 이후 체간에 발생하는 양성 혈관 증식이다.
화농성 육아종은 빠르게 성장하는 붉은 결절로, 외상 후 발생하며 출혈이 잦다.
대부분 양성이나, 드물게 악성 혈관종양과의 감별이 필요하다.""",
    },
    {
        "doc_id": "dermoscopy_basics",
        "title": "더모스코피 기본 패턴",
        "category": "general",
        "content": """더모스코피(피부확대경)는 피부 병변의 미세구조를 관찰하는 비침습적 진단 도구이다.
주요 감별 패턴:
멜라노마: 비정형 색소 네트워크, 비정형 점/소구, 청백색 구조, 불규칙 줄무늬.
기저세포암: 나뭇가지 모양 혈관, 잎사귀 모양 구조, 청회색 난원형 집.
지루각화증: 뇌회전 패턴, 뿔 낭종, 벌레먹은 경계.
모반: 규칙적 색소 네트워크, 대칭 패턴, 균일한 소구.
혈관병변: 적색-청색 라군, 규칙적 혈관 패턴.
2-step 알고리즘: 1단계 멜라닌세포성 여부 → 2단계 양성/악성 감별.""",
    },
    {
        "doc_id": "referral_guidelines",
        "title": "전문의 의뢰 기준",
        "category": "general",
        "content": """다음 조건 중 하나 이상 해당 시 피부과 전문의 의뢰를 권고한다:
1. ABCDE 기준 중 2개 이상 해당하는 병변.
2. 최근 6개월 이내 크기, 색상, 모양이 변한 병변.
3. 궤양, 출혈, 가려움이 동반된 기존 병변.
4. 50세 이상에서 새로 발생한 색소성 병변.
5. 환자 또는 가족이 멜라노마 병력이 있는 경우.
6. 면역억제 상태의 환자에서 발생한 새 병변.
긴급 의뢰 (2주 이내): 멜라노마 의심, 빠르게 성장하는 병변.
일반 의뢰 (6주 이내): 진단 불확실, 감별이 필요한 경우.""",
    },
]


# ============================================================
# 6. 통합 문서 로더
# ============================================================
def load_all_documents(rag_dir: Path) -> List[DocumentChunk]:
    """
    모든 소스에서 문서를 로드하고 청킹하여 통합 리스트 반환.
    우선순위: PDF → 텍스트 파일 → 시드 문서(항상 포함)
    """
    all_chunks: List[DocumentChunk] = []
    pdf_count, txt_count = 0, 0

    # --- PDF 파일 처리 ---
    pdf_files = list(rag_dir.glob("**/*.pdf"))
    if pdf_files:
        print(f"\n[PDF 로드] {len(pdf_files)}개 파일 발견")
        for pdf_path in sorted(pdf_files):
            pages = extract_text_from_pdf(pdf_path)
            if not pages:
                continue

            doc_id = pdf_path.stem.lower().replace(" ", "_")
            full_text = "\n\n".join([p["text"] for p in pages])
            chunks = chunk_documents(full_text)

            for i, chunk_text in enumerate(chunks):
                page_num = 1
                for p in pages:
                    if chunk_text[:50] in p["text"]:
                        page_num = p["page"]
                        break

                all_chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    title=pdf_path.stem,
                    source=pdf_path.name,
                    source_type="pdf",
                    category=infer_category(chunk_text, pdf_path.stem),
                    page_number=page_num,
                    chunk_index=i,
                    content=chunk_text,
                ))
            pdf_count += 1

    # --- 텍스트 파일 처리 ---
    for ext in ["**/*.txt", "**/*.md"]:
        for txt_path in sorted(rag_dir.glob(ext)):
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            if len(text.strip()) < 50:
                continue

            doc_id = txt_path.stem.lower().replace(" ", "_")
            chunks = chunk_documents(text)

            for i, chunk_text in enumerate(chunks):
                all_chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    title=txt_path.stem,
                    source=txt_path.name,
                    source_type="txt",
                    category=infer_category(chunk_text, txt_path.stem),
                    page_number=0,
                    chunk_index=i,
                    content=chunk_text,
                ))
            txt_count += 1

    # --- 시드 문서 (항상 포함) ---
    print(f"\n[시드 문서] {len(SEED_DOCUMENTS)}개 로드")
    for doc in SEED_DOCUMENTS:
        chunks = chunk_documents(doc["content"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append(DocumentChunk(
                doc_id=doc["doc_id"],
                title=doc["title"],
                source="seed_document",
                source_type="seed",
                category=doc["category"],
                page_number=0,
                chunk_index=i,
                content=chunk_text,
            ))

    print(f"\n[통합] PDF: {pdf_count}, TXT: {txt_count}, "
          f"시드: {len(SEED_DOCUMENTS)} → 총 청크: {len(all_chunks)}개")

    return all_chunks


# ============================================================
# 7. 임베딩 생성
# ============================================================
def load_embedding_model():
    """
    paraphrase-multilingual-MiniLM-L12-v2 — 한국어 포함 50개+ 언어 지원.
    config.py의 all-MiniLM-L6-v2(영어 전용) 대신 다국어 모델 사용.
    """
    from sentence_transformers import SentenceTransformer
    model_name = "BAAI/bge-m3"
    model = SentenceTransformer(model_name)
    print(f"[임베딩] {model_name} (차원: {model.get_sentence_embedding_dimension()})")
    return model


def generate_embeddings(model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    embeddings = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ============================================================
# 8. SQLite 벡터 DB 구축
# ============================================================
def build_vector_db(chunks: List[DocumentChunk], embeddings: np.ndarray, db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            doc_id TEXT NOT NULL,
            title TEXT,
            source TEXT,
            source_type TEXT,
            category TEXT,
            page_number INTEGER,
            chunk_index INTEGER,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX idx_category ON documents(category)")

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        cursor.execute(
            """INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (i, chunk.doc_id, chunk.title, chunk.source, chunk.source_type,
             chunk.category, chunk.page_number, chunk.chunk_index,
             chunk.content, emb.tobytes()),
        )

    conn.commit()
    conn.close()

    db_size = db_path.stat().st_size / 1024 / 1024
    src_stats = {}
    for c in chunks:
        src_stats[c.source_type] = src_stats.get(c.source_type, 0) + 1
    print(f"[DB] {db_path} ({db_size:.2f} MB)")
    print(f"  청크: {len(chunks)}개, 차원: {embeddings.shape[1]}, 소스: {src_stats}")


# ============================================================
# 9. 검색 함수 (추론 시에도 import하여 재사용)
# ============================================================
def search_rag(
    db_path: Path,
    query_embedding: np.ndarray,
    top_k: int = 5,
    category_filter: Optional[str] = None,
) -> List[Dict]:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    if category_filter:
        cursor.execute(
            "SELECT id, doc_id, title, source, source_type, category, "
            "page_number, content, embedding FROM documents WHERE category = ?",
            (category_filter,)
        )
    else:
        cursor.execute(
            "SELECT id, doc_id, title, source, source_type, category, "
            "page_number, content, embedding FROM documents"
        )

    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return []

    db_embeddings = np.array([np.frombuffer(r[8], dtype=np.float32) for r in rows])
    similarities = (db_embeddings @ query_embedding.reshape(-1, 1)).squeeze()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        r = rows[idx]
        results.append({
            "id": r[0], "doc_id": r[1], "title": r[2],
            "source": r[3], "source_type": r[4], "category": r[5],
            "page_number": r[6], "content": r[7],
            "similarity": float(similarities[idx]),
        })
    return results


# ============================================================
# 10. 검색 테스트 (정확도 자동 측정)
# ============================================================
def run_search_tests(db_path: Path, embedding_model):
    test_queries = [
        ("이 점이 멜라노마인지 어떻게 알 수 있나요?", "mel"),
        ("피부에 진주빛 결절이 생겼는데 위험한가요?", "bcc"),
        ("ABCDE 규칙이 뭔가요?", "general"),
        ("언제 병원에 가야 하나요?", "general"),
        ("혈관종은 악성인가요?", "vasc"),
        ("피부가 사포처럼 거칠어졌어요", "akiec"),
        ("갑자기 점이 많이 생겼어요", "bkl"),
    ]

    print("\n" + "=" * 60)
    print(" RAG 검색 테스트")
    print("=" * 60)

    correct = 0
    for query, expected_cat in test_queries:
        query_emb = embedding_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)[0]

        results = search_rag(db_path, query_emb, top_k=3)
        top1_cat = results[0]["category"] if results else "없음"
        match = "O" if top1_cat == expected_cat else "X"
        if match == "O":
            correct += 1

        print(f"\n[{match}] Query: {query}")
        print(f"    기대: {expected_cat} | Top-1: {top1_cat}")
        for rank, r in enumerate(results, 1):
            src = f"[{r['source_type']}]" if r['source_type'] != 'seed' else '[seed]'
            print(f"    #{rank} (sim={r['similarity']:.4f}) "
                  f"[{r['category']}] {r['title']} {src}")

    accuracy = correct / len(test_queries) * 100
    print(f"\n[검색 정확도] {correct}/{len(test_queries)} ({accuracy:.0f}%)")
    if accuracy < 70:
        print("  → 70% 미만: 지식베이스 문서 보강이 필요합니다.")
        print("  → data/rag_knowledge/ 에 관련 PDF를 추가하세요.")


# ============================================================
# 11. 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print(" [1주차] RAG 지식베이스 구축 (v2 — PDF 지원)")
    print("=" * 60)

    RAG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_count = len(list(RAG_DIR.glob("**/*.pdf")))
    txt_count = len(list(RAG_DIR.glob("**/*.txt"))) + len(list(RAG_DIR.glob("**/*.md")))
    print(f"\n[소스 디렉터리] {RAG_DIR}")
    print(f"  PDF: {pdf_count}개, 텍스트: {txt_count}개")

    if pdf_count == 0 and txt_count == 0:
        print("\n  [안내] 외부 문서가 없어 시드 문서 + PDF 추가를 권장합니다.")
        print("  추천 PDF:")
        print("    - WHO Skin NTDs: who.int/publications")
        print("    - BAD Guidelines: bad.org.uk/clinical-standards")
        print("    - DermNet NZ: dermnetnz.org (웹 → txt 저장)")

    all_chunks = load_all_documents(RAG_DIR)

    print("\n[임베딩 생성]")
    embedding_model = load_embedding_model()
    texts = [c.content for c in all_chunks]
    embeddings = generate_embeddings(embedding_model, texts)

    print("\n[DB 구축]")
    db_path = RAG_DB_DIR / "medical_knowledge.db"
    build_vector_db(all_chunks, embeddings, db_path)

    run_search_tests(db_path, embedding_model)

    # 가이드 저장
    guide = {
        "db_path": str(db_path),
        "총_청크": len(all_chunks),
        "임베딩_모델": "paraphrase-multilingual-MiniLM-L12-v2",
        "소스_통계": {
            st: sum(1 for c in all_chunks if c.source_type == st)
            for st in set(c.source_type for c in all_chunks)
        },
        "PDF_추가_방법": "data/rag_knowledge/ 에 PDF를 넣고 재실행",
    }
    guide_path = RAG_DB_DIR / "knowledge_base_guide.json"
    with open(guide_path, "w", encoding="utf-8") as f:
        json.dump(guide, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 60)
    print(f" 완료. DB: {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()