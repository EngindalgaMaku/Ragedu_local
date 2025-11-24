import requests
import json

session_id = "7e1c08925280362b75af7cae3b561a6a"

print(f"🔍 Testing chunks for session: {session_id}\n")

# Get chunks
r = requests.get(f"http://localhost:8003/sessions/{session_id}/chunks", timeout=30)
if r.status_code != 200:
    print(f"❌ Error: {r.status_code}")
    print(r.text)
    exit(1)

chunks = r.json().get("chunks", [])
print(f"✅ Found {len(chunks)} chunks\n")

if len(chunks) < 2:
    print("⚠️ Need at least 2 chunks for testing")
    exit(1)

# Test overlap
print("="*80)
print("OVERLAP TEST:")
print("="*80)

issues = []
for i in range(len(chunks) - 1):
    current = chunks[i]
    next_chunk = chunks[i + 1]
    
    current_text = current.get("chunk_text", "").strip()
    next_text = next_chunk.get("chunk_text", "").strip()
    
    current_idx = current.get("chunk_index", i + 1)
    next_idx = next_chunk.get("chunk_index", i + 2)
    
    # Check last 100 chars vs first 100 chars
    current_end = current_text[-100:] if len(current_text) > 100 else current_text
    next_start = next_text[:100] if len(next_text) > 100 else next_text
    
    if current_end.strip() == next_start.strip():
        issues.append((current_idx, next_idx, "EXACT_DUPLICATE"))
    elif current_end.strip() in next_text or next_start.strip() in current_text:
        issues.append((current_idx, next_idx, "PARTIAL_OVERLAP"))

if issues:
    print(f"\n❌ {len(issues)} OVERLAP ISSUES FOUND:\n")
    for idx1, idx2, issue_type in issues[:10]:
        print(f"   Chunk {idx1} -> {idx2}: {issue_type}")
else:
    print("\n✅ No overlap issues found!")

# Show first 3 chunks
print(f"\n{'='*80}")
print("FIRST 3 CHUNKS:")
print("="*80)

for i in range(min(3, len(chunks))):
    chunk = chunks[i]
    text = chunk.get("chunk_text", "")
    idx = chunk.get("chunk_index", i + 1)
    
    print(f"\n--- Chunk {idx} (Length: {len(text)}) ---")
    print(f"Start: {text[:150]}...")
    print(f"End: ...{text[-150:]}")



