"""
Build FinRAG tree once using all PDFs in the data folder.
This script should be run once to create the tree, then all other scripts can load it.
"""
import os
import sys
import json
import re
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables from .env file
from finrag.utils import load_env_file
load_env_file()

from finrag import FinRAGConfig, FinRAG
from finrag.vectorstore.annual_report_store import ANNUAL_REPORT_TREE_PATH
from finrag.vectorstore.chroma_store import ChromaConfig, ChromaVectorStore


def _report_document(pdf_path: Path, mapping_path: Path) -> str:
    """Add reliable document identity headers from the shared ticker mapping."""
    normalized_filename = re.sub(r"[^A-Z0-9]+", " ", pdf_path.stem.upper()).strip()
    try:
        mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load ticker mapping {mapping_path}: {exc}") from exc

    matches = []
    for item in mappings:
        ticker = item.get("ticker", "")
        if not ticker.endswith((".NS", ".BO")):
            continue
        ticker_stem = ticker.rsplit(".", 1)[0].upper()
        aliases = [item.get("company_name", ""), *item.get("aliases", [])]
        matching_alias_lengths = [
            len(normalized_alias)
            for alias in aliases
            if alias
            for normalized_alias in [re.sub(r"[^A-Z0-9]+", " ", alias.upper()).strip()]
            if re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_filename)
        ]
        ticker_match = bool(re.match(rf"^{re.escape(ticker_stem)}(?:\b|$)", normalized_filename))
        if matching_alias_lengths or ticker_match:
            matches.append((max(matching_alias_lengths, default=1 if ticker_match else 0), item))

    lines = [f"Source: {pdf_path.name}", "Document Type: annual_report"]
    if matches:
        record = sorted(matches, key=lambda candidate: candidate[0], reverse=True)[0][1]
        lines.extend((f"Company: {record['company_name']}", f"Ticker: {record['ticker']}"))
        year = re.search(r"(?:19|20)\d{2}[_-](?:\d{2}|(?:19|20)\d{2})", pdf_path.stem)
        if year:
            lines.append(f"Financial Year: {year.group(0).replace('_', '-')}" )
    return "\n".join(lines) + "\n"


def build_tree_from_all_pdfs():
    """Build FinRAG tree using all PDFs in the data folder."""
    
    # Initialize config
    config = FinRAGConfig()
    config.use_metadata_clustering = True  # Enable metadata-based clustering

    existing_index_path = ANNUAL_REPORT_TREE_PATH / "chroma_vectorstore"
    if existing_index_path.is_dir():
        try:
            existing_store = ChromaVectorStore(ChromaConfig(
                collection_name=config.chroma_collection_name,
                persist_directory=str(existing_index_path),
            ))
            existing_count = existing_store.collection.count()
        except Exception as exc:
            print(f"Cannot safely inspect existing ChromaDB index: {exc}")
            print("Stopping before indexing to preserve the existing database.")
            return False
        if existing_count:
            print(
                f"Annual report index already contains {existing_count} vectors at "
                f"{existing_index_path}; skipping duplicate indexing."
            )
            return True
    
    # Initialize FinRAG
    print("=" * 80)
    print("BUILDING FINRAG TREE FROM ALL PDFs")
    print("=" * 80)
    print("\nInitializing FinRAG system...")
    finrag = FinRAG(config)
    
    # Get all PDFs from data folder
    data_dir = Path(__file__).parent.parent / "data"
    mapping_path = data_dir / "mappings" / "ticker_mapping.json"
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in {data_dir}")
        print("Please add PDF files to the data folder and try again.")
        return False
    
    print(f"\n✓ Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Process all PDFs
    all_documents = []
    print("\n" + "=" * 80)
    print("PROCESSING PDFs")
    print("=" * 80)
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print("-" * 80)
        
        try:
            # Load PDF with filtering if enabled
            use_filtering = getattr(config, 'use_filtered_parsing', False)
            if use_filtering:
                print("  Using filtered parsing to extract key sections...")
            
            text = finrag.load_pdf(str(pdf_path))
            print(f"  ✓ Loaded {len(text):,} characters")
            
            all_documents.append(_report_document(pdf_path, mapping_path) + text)
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")
            print(f"     Skipping this file...")
            continue
    
    if not all_documents:
        print("\n❌ No documents were successfully processed!")
        return False
    
    print(f"\n✓ Successfully processed {len(all_documents)}/{len(pdf_files)} PDFs")
    
    # Build the tree
    print("\n" + "=" * 80)
    print("BUILDING RAPTOR TREE")
    print("=" * 80)
    print("\nThis may take several minutes depending on the number of documents...")
    
    try:
        finrag.add_documents(all_documents)
        print("\n✓ Tree built successfully!")
    except Exception as e:
        print(f"\n❌ Error building tree: {e}")
        traceback.print_exc()
        return False
    
    # Print statistics
    print("\n" + "=" * 80)
    print("TREE STATISTICS")
    print("=" * 80)
    
    stats = finrag.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save the tree
    save_path = ANNUAL_REPORT_TREE_PATH
    print("\n" + "=" * 80)
    print("SAVING TREE")
    print("=" * 80)
    print(f"\nSaving to: {save_path}")
    
    try:
        finrag.save(str(save_path))
        print(f"✓ Tree saved successfully to {save_path}")
    except Exception as e:
        print(f"❌ Error saving tree: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("BUILD COMPLETE!")
    print("=" * 80)
    print(f"""
✓ Processed {len(all_documents)} documents
✓ Built RAPTOR tree with {stats.get('total_nodes', 'N/A')} nodes
✓ Tree saved to: {save_path}

You can now use this tree in your applications by loading it:

    from finrag import FinRAG, FinRAGConfig
    
    config = FinRAGConfig()
    finrag = FinRAG(config)
    finrag.load("{save_path}")
    
    # Now query the system
    result = finrag.query("Your question here")

To rebuild the tree with updated PDFs, run this script again:
    python scripts/build_tree.py
""")
    
    return True


if __name__ == "__main__":
    success = build_tree_from_all_pdfs()
    sys.exit(0 if success else 1)
