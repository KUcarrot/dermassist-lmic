"""Build RAG knowledge base from medical sources."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dermassist.rag.build_db import main

if __name__ == "__main__":
    main()
