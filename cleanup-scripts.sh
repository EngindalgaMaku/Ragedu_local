#!/bin/bash

echo "🧹 CLEANUP - Test scriptlerini tests klasörüne taşı"
echo "================================================"

# Create tests directory if not exists
mkdir -p rag3_for_local/tests/debug_scripts

echo "📁 Test scriptleri taşınıyor..."

# Move all debug/test scripts to tests folder
mv rag3_for_local/AUTH-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/FINAL-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/API-GATEWAY-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/QUICK-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/DEBUG-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/CREDENTIALS-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/EMERGENCY-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/URGENT-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/FORCE-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/PATCH-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/FIX-*.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null
mv rag3_for_local/git-pull-fix.sh rag3_for_local/tests/debug_scripts/ 2>/dev/null

# Move test_*.py files that are not in tests/ already
mv rag3_for_local/test_*.py rag3_for_local/tests/ 2>/dev/null
mv rag3_for_local/check_*.py rag3_for_local/tests/ 2>/dev/null

# Keep essential files in root
echo "📋 Ana dizinde kalacak dosyalar:"
ls -la rag3_for_local/ | grep -E "(docker-compose|Dockerfile|README|requirements|\.env|\.py$)" | head -10

echo
echo "🗑️  Taşınan debug scriptleri:"
ls -la rag3_for_local/tests/debug_scripts/ | head -10

echo
echo "✅ Cleanup tamamlandı!"
echo "📁 Debug scriptleri: tests/debug_scripts/"
echo "🧪 Test dosyaları: tests/"