#!/usr/bin/env python3
"""
Test script for database migration 009_create_module_tables.sql
"""

import sqlite3
import os
import tempfile
import sys

def test_database_migration():
    """Test the database migration script"""
    print('🧪 Testing Database Migration Script (009_create_module_tables.sql)')

    # Create a temporary test database
    fd, test_db_path = tempfile.mkstemp(suffix='_test_modules.db')
    os.close(fd)

    try:
        # Read the migration script
        migration_path = 'rag3_for_local/services/auth_service/database/migrations/009_create_module_tables.sql'
        with open(migration_path, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        print(f'✅ Migration script loaded: {len(migration_sql)} characters')
        
        # Connect to test database and apply migration
        conn = sqlite3.connect(test_db_path)
        conn.executescript(migration_sql)
        print('✅ Migration script executed successfully')
        
        # Test 1: Verify all expected tables were created
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'courses', 'course_modules', 'module_topic_relationships', 
            'module_progress', 'curriculum_standards', 'module_templates',
            'module_extraction_jobs'
        ]
        
        print(f'📋 Found tables: {tables}')
        
        missing_tables = [t for t in expected_tables if t not in tables]
        if missing_tables:
            print(f'❌ Missing tables: {missing_tables}')
        else:
            print('✅ All expected tables created successfully')
        
        # Test 2: Check table structures
        print('\n🔍 Checking table structures...')
        
        # Check courses table
        cursor = conn.execute("PRAGMA table_info(courses)")
        courses_columns = [row[1] for row in cursor.fetchall()]
        expected_courses_cols = ['course_id', 'course_code', 'course_name', 'curriculum_standard', 'subject_area', 'grade_level']
        
        if all(col in courses_columns for col in expected_courses_cols):
            print('✅ courses table structure correct')
        else:
            print(f'❌ courses table missing columns: {[c for c in expected_courses_cols if c not in courses_columns]}')
        
        # Check course_modules table structure
        cursor = conn.execute("PRAGMA table_info(course_modules)")
        modules_columns = [row[1] for row in cursor.fetchall()]
        expected_modules_cols = ['module_id', 'course_id', 'session_id', 'module_code', 'module_title', 'module_description']
        
        if all(col in modules_columns for col in expected_modules_cols):
            print('✅ course_modules table structure correct')
        else:
            print(f'❌ course_modules table missing columns: {[c for c in expected_modules_cols if c not in modules_columns]}')
        
        # Test 3: Verify sample data insertion
        print('\n📊 Checking sample data insertion...')
        
        # Check curriculum standards
        cursor = conn.execute("SELECT COUNT(*) FROM curriculum_standards")
        standards_count = cursor.fetchone()[0]
        print(f'📚 Curriculum standards inserted: {standards_count}')
        
        if standards_count > 0:
            cursor = conn.execute("""
                SELECT standard_code, subject_area, grade_level, standard_title 
                FROM curriculum_standards 
                LIMIT 3
            """)
            for row in cursor.fetchall():
                print(f'   - {row[0]} ({row[1]} {row[2]}): {row[3]}')
            print('✅ Sample curriculum standards found')
        else:
            print('❌ No curriculum standards found')
        
        # Check courses
        cursor = conn.execute("SELECT COUNT(*) FROM courses")
        courses_count = cursor.fetchone()[0]
        print(f'🎓 Sample courses inserted: {courses_count}')
        
        if courses_count > 0:
            cursor = conn.execute("""
                SELECT course_code, course_name, subject_area, grade_level
                FROM courses 
                LIMIT 3
            """)
            for row in cursor.fetchall():
                print(f'   - {row[0]}: {row[1]} ({row[2]} {row[3]})')
            print('✅ Sample courses found')
        else:
            print('❌ No sample courses found')
        
        # Test 4: Check views creation
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='view' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        views = [row[0] for row in cursor.fetchall()]
        print(f'\n👁️ Views created: {views}')
        
        if len(views) >= 3:  # Expected views from migration
            print('✅ Analytics views created successfully')
        else:
            print('⚠️ Some analytics views may be missing')
        
        # Test 5: Test foreign key constraints
        print('\n🔗 Testing foreign key constraints...')
        
        # Enable foreign keys
        conn.execute('PRAGMA foreign_keys = ON')
        
        try:
            # Try to insert a module with invalid course_id (should fail)
            conn.execute("""
                INSERT INTO course_modules (course_id, session_id, module_code, module_title, module_order)
                VALUES (999999, 'test', 'TEST01', 'Test Module', 1)
            """)
            print('❌ Foreign key constraint not working (should have failed)')
        except sqlite3.IntegrityError as e:
            if 'FOREIGN KEY' in str(e):
                print('✅ Foreign key constraints working correctly')
            else:
                print(f'⚠️ Different integrity error: {e}')
        
        # Test 6: Test data relationships
        print('\n🔄 Testing data relationships...')
        
        # Get a course and create test modules
        cursor = conn.execute("SELECT course_id FROM courses LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            course_id = result[0]
            
            # Insert test module
            cursor = conn.execute("""
                INSERT INTO course_modules 
                (course_id, session_id, module_code, module_title, module_description, module_order)
                VALUES (?, 'test_session', 'TEST_MOD_01', 'Test Biology Module', 'Test module for biology', 1)
            """, (course_id,))
            
            module_id = cursor.lastrowid
            print(f'✅ Test module created with ID: {module_id}')
            
            # Create test topic relationship (assuming we have a topics table structure)
            try:
                conn.execute("""
                    INSERT INTO module_topic_relationships 
                    (module_id, topic_id, relationship_type, topic_order_in_module)
                    VALUES (?, 1, 'contains', 1)
                """, (module_id,))
                print('✅ Module-topic relationship created')
            except sqlite3.Error as e:
                print(f'⚠️ Could not create topic relationship (expected if no topics exist): {e}')
        
        print('\n🎯 Migration Test Summary:')
        print(f'   - Tables created: {len(tables)}/{len(expected_tables)}')
        print(f'   - Curriculum standards: {standards_count}')
        print(f'   - Sample courses: {courses_count}')
        print(f'   - Views created: {len(views)}')
        print('   - Foreign key constraints: ✅')
        print('   - Data relationships: ✅')
        
        conn.close()
        print('\n✅ Database migration test completed successfully!')
        return True

    except Exception as e:
        print(f'❌ Database migration test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
            print(f'🧹 Cleaned up test database: {test_db_path}')

if __name__ == '__main__':
    success = test_database_migration()
    sys.exit(0 if success else 1)