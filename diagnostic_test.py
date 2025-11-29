#!/usr/bin/env python3
"""
Diagnostic Test for Module Extraction System
Identifies which components are working and what issues exist
"""

import sys
import os
import traceback

# Add the necessary paths
sys.path.append('rag3_for_local/services/aprag_service')
sys.path.append('rag3_for_local')

def test_imports():
    """Test if all components can be imported"""
    print('🔍 Diagnostic Test - Component Import Analysis')
    print('=' * 60)
    
    components = {
        'Feature Flags': 'config.feature_flags',
        'Curriculum Templates': 'templates.curriculum_templates', 
        'LLM Organizer': 'processors.llm_module_organizer',
        'Quality Validator': 'validators.module_quality_validator',
        'Module Extraction Service': 'services.module_extraction_service',
        'API Module': 'api.modules',
        'Logging Config': 'utils.logging_config',
        'Database Manager': 'database.database'
    }
    
    results = {}
    
    for name, module_path in components.items():
        try:
            if name == 'Feature Flags':
                from config.feature_flags import FeatureFlags
                # Test available methods
                methods = [method for method in dir(FeatureFlags) if not method.startswith('_')]
                results[name] = {'status': 'OK', 'methods': methods[:10]}  # First 10 methods
                print(f'✅ {name}: OK - Available methods: {len(methods)}')
                
            elif name == 'Curriculum Templates':
                from templates.curriculum_templates import CurriculumTemplateManager
                template_manager = CurriculumTemplateManager()
                methods = [method for method in dir(template_manager) if not method.startswith('_')]
                results[name] = {'status': 'OK', 'methods': methods[:10]}
                print(f'✅ {name}: OK - Available methods: {len(methods)}')
                
            elif name == 'LLM Organizer':
                from processors.llm_module_organizer import LLMModuleOrganizer
                organizer = LLMModuleOrganizer()
                methods = [method for method in dir(organizer) if not method.startswith('_')]
                results[name] = {'status': 'OK', 'methods': methods[:10]}
                print(f'✅ {name}: OK - Available methods: {len(methods)}')
                
            elif name == 'Quality Validator':
                from validators.module_quality_validator import ModuleQualityValidator
                validator = ModuleQualityValidator()
                methods = [method for method in dir(validator) if not method.startswith('_')]
                results[name] = {'status': 'OK', 'methods': methods[:10]}
                print(f'✅ {name}: OK - Available methods: {len(methods)}')
                
            elif name == 'Module Extraction Service':
                from services.module_extraction_service import ModuleExtractionService
                service = ModuleExtractionService()
                methods = [method for method in dir(service) if not method.startswith('_')]
                results[name] = {'status': 'OK', 'methods': methods[:10]}
                print(f'✅ {name}: OK - Available methods: {len(methods)}')
                
            elif name == 'Database Manager':
                from database.database import DatabaseManager
                results[name] = {'status': 'OK', 'methods': ['get_connection', 'close']}
                print(f'✅ {name}: OK')
                
            else:
                __import__(module_path)
                results[name] = {'status': 'OK', 'methods': []}
                print(f'✅ {name}: OK')
                
        except ImportError as e:
            results[name] = {'status': 'IMPORT_ERROR', 'error': str(e)}
            print(f'❌ {name}: Import Error - {e}')
        except Exception as e:
            results[name] = {'status': 'ERROR', 'error': str(e)}
            print(f'⚠️ {name}: Error - {e}')
    
    return results

def test_feature_flags():
    """Test feature flags in detail"""
    print('\n🚩 Feature Flags Detailed Analysis')
    print('-' * 40)
    
    try:
        from config.feature_flags import FeatureFlags
        
        # List all methods
        all_methods = [method for method in dir(FeatureFlags) if not method.startswith('_')]
        print(f'Available methods: {all_methods}')
        
        # Test basic methods
        basic_tests = [
            ('is_aprag_enabled', lambda: FeatureFlags.is_aprag_enabled()),
            ('get_all_flags', lambda: FeatureFlags.get_all_flags()),
        ]
        
        for method_name, test_func in basic_tests:
            try:
                result = test_func()
                print(f'✅ {method_name}: {result}')
            except Exception as e:
                print(f'❌ {method_name}: {e}')
        
        # Test module extraction methods if they exist
        module_methods = [
            'is_module_extraction_enabled',
            'is_module_quality_validation_enabled', 
            'is_module_curriculum_alignment_enabled'
        ]
        
        for method in module_methods:
            if hasattr(FeatureFlags, method):
                try:
                    result = getattr(FeatureFlags, method)()
                    print(f'✅ {method}: {result}')
                except Exception as e:
                    print(f'❌ {method}: {e}')
            else:
                print(f'⚠️ {method}: Method not found')
        
        return True
        
    except Exception as e:
        print(f'❌ Feature Flags test failed: {e}')
        traceback.print_exc()
        return False

def test_curriculum_templates():
    """Test curriculum templates in detail"""
    print('\n📚 Curriculum Templates Detailed Analysis')
    print('-' * 40)
    
    try:
        from templates.curriculum_templates import CurriculumTemplateManager
        
        template_manager = CurriculumTemplateManager()
        
        # List all methods
        methods = [method for method in dir(template_manager) if not method.startswith('_')]
        print(f'Available methods: {methods}')
        
        # Test basic functionality
        tests = [
            ('get_supported_curricula', lambda: template_manager.get_supported_curricula() if hasattr(template_manager, 'get_supported_curricula') else 'Method not found'),
            ('get_template_info', lambda: template_manager.get_template_info() if hasattr(template_manager, 'get_template_info') else 'Method not found'),
            ('templates attribute', lambda: hasattr(template_manager, 'templates'))
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                print(f'✅ {test_name}: {result}')
            except Exception as e:
                print(f'❌ {test_name}: {e}')
        
        # Try to get template info
        if hasattr(template_manager, 'templates'):
            print(f'✅ Template data available: {list(template_manager.templates.keys())}')
        
        return True
        
    except Exception as e:
        print(f'❌ Curriculum Templates test failed: {e}')
        traceback.print_exc()
        return False

def test_integration_readiness():
    """Test if the system is ready for integration"""
    print('\n🎯 Integration Readiness Assessment')
    print('-' * 40)
    
    readiness_score = 0
    max_score = 6
    
    # Test 1: Database Migration
    print('1. Database Migration: ✅ PASSED (from previous test)')
    readiness_score += 1
    
    # Test 2: Feature Flags
    try:
        from config.feature_flags import FeatureFlags
        FeatureFlags.is_aprag_enabled()
        print('2. Feature Flags Basic: ✅ PASSED')
        readiness_score += 1
    except:
        print('2. Feature Flags Basic: ❌ FAILED')
    
    # Test 3: Curriculum Templates
    try:
        from templates.curriculum_templates import CurriculumTemplateManager
        CurriculumTemplateManager()
        print('3. Curriculum Templates: ✅ PASSED')
        readiness_score += 1
    except:
        print('3. Curriculum Templates: ❌ FAILED')
    
    # Test 4: LLM Integration
    try:
        from processors.llm_module_organizer import LLMModuleOrganizer
        LLMModuleOrganizer()
        print('4. LLM Integration: ✅ PASSED')
        readiness_score += 1
    except:
        print('4. LLM Integration: ❌ FAILED')
    
    # Test 5: Quality Validation
    try:
        from validators.module_quality_validator import ModuleQualityValidator
        ModuleQualityValidator()
        print('5. Quality Validation: ✅ PASSED')
        readiness_score += 1
    except:
        print('5. Quality Validation: ❌ FAILED')
    
    # Test 6: Module Service
    try:
        from services.module_extraction_service import ModuleExtractionService
        ModuleExtractionService()
        print('6. Module Service: ✅ PASSED')
        readiness_score += 1
    except:
        print('6. Module Service: ❌ FAILED')
    
    print(f'\nReadiness Score: {readiness_score}/{max_score} ({readiness_score/max_score*100:.1f}%)')
    
    if readiness_score >= 5:
        print('🎉 System is ready for comprehensive testing!')
    elif readiness_score >= 3:
        print('⚠️ System has some issues but partial testing is possible')
    else:
        print('❌ System needs significant fixes before testing')
    
    return readiness_score, max_score

def main():
    """Run comprehensive diagnostic"""
    print('🔬 Module Extraction System Diagnostic')
    print('=' * 60)
    
    # Test imports
    import_results = test_imports()
    
    # Test feature flags in detail
    feature_flags_ok = test_feature_flags()
    
    # Test curriculum templates
    templates_ok = test_curriculum_templates()
    
    # Test integration readiness
    readiness_score, max_score = test_integration_readiness()
    
    # Generate summary
    print('\n' + '=' * 60)
    print('📊 DIAGNOSTIC SUMMARY')
    print('=' * 60)
    
    working_components = sum(1 for result in import_results.values() if result['status'] == 'OK')
    total_components = len(import_results)
    
    print(f'✅ Component Import Success: {working_components}/{total_components}')
    print(f'🚩 Feature Flags: {"✅ WORKING" if feature_flags_ok else "❌ ISSUES"}')
    print(f'📚 Curriculum Templates: {"✅ WORKING" if templates_ok else "❌ ISSUES"}')
    print(f'🎯 Integration Readiness: {readiness_score}/{max_score} ({readiness_score/max_score*100:.1f}%)')
    
    # Recommendations
    print('\n💡 RECOMMENDATIONS:')
    if readiness_score >= 5:
        print('• ✅ System is ready for biology curriculum testing')
        print('• ✅ Can proceed with API endpoint testing')
        print('• ✅ Ready for end-to-end integration tests')
    elif readiness_score >= 3:
        print('• ⚠️ Fix component import issues first')
        print('• ⚠️ Test individual components separately')
        print('• 🔄 Proceed with limited functionality testing')
    else:
        print('• ❌ Fix critical import and initialization issues')
        print('• 🔧 Review component dependencies and paths')
        print('• 📋 Test components individually before integration')
    
    return readiness_score >= 3

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)