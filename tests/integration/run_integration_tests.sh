#!/bin/bash

# Integration Tests Runner Script for RAG Education Assistant
# Comprehensive testing of authentication system and components

set -e  # Exit on any error

echo "🚀 RAG Education Assistant - Integration Tests Runner"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AUTH_SERVICE_PORT=8002
API_GATEWAY_PORT=8000
FRONTEND_PORT=3000
TEST_TIMEOUT=300  # 5 minutes
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Service URLs
AUTH_SERVICE_URL="http://localhost:${AUTH_SERVICE_PORT}"
API_GATEWAY_URL="http://localhost:${API_GATEWAY_PORT}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"

# Test categories
TEST_CATEGORIES=("database" "auth_system" "api_endpoints" "frontend_auth")

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if a service is running
check_service() {
    local url=$1
    local service_name=$2
    local timeout=30
    local count=0
    
    print_color $YELLOW "Checking ${service_name} at ${url}..."
    
    while [ $count -lt $timeout ]; do
        if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
            print_color $GREEN "✅ ${service_name} is running"
            return 0
        fi
        sleep 2
        count=$((count + 2))
        printf "."
    done
    
    echo ""
    print_color $RED "❌ ${service_name} is not responding at ${url}"
    return 1
}

# Function to wait for services to be ready
wait_for_services() {
    print_color $BLUE "🔍 Checking required services..."
    echo ""
    
    local services_ok=true
    
    # Check Auth Service
    if ! check_service "${AUTH_SERVICE_URL}/health" "Auth Service"; then
        print_color $YELLOW "⚠️  Auth Service not running. Some tests will be skipped."
        print_color $YELLOW "   Start with: python services/auth_service/main.py"
        echo ""
    fi
    
    # Check API Gateway
    if ! check_service "${API_GATEWAY_URL}/health" "API Gateway"; then
        print_color $YELLOW "⚠️  API Gateway not running. Some tests will be skipped."
        print_color $YELLOW "   Start with: python src/api/main.py"
        echo ""
    fi
    
    # Check Frontend
    if ! check_service "${FRONTEND_URL}" "Frontend"; then
        print_color $YELLOW "⚠️  Frontend not running. UI tests will be skipped."
        print_color $YELLOW "   Start with: cd frontend && npm run dev"
        echo ""
    fi
    
    echo ""
}

# Function to setup Python environment
setup_python_env() {
    print_color $BLUE "🐍 Setting up Python environment..."
    
    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        print_color $YELLOW "⚠️  Not in a virtual environment. Creating one..."
        
        if ! command -v python3 &> /dev/null; then
            print_color $RED "❌ Python 3 not found. Please install Python 3."
            exit 1
        fi
        
        python3 -m venv venv
        source venv/bin/activate
        print_color $GREEN "✅ Virtual environment created and activated"
    fi
    
    # Install testing dependencies
    print_color $BLUE "📦 Installing test dependencies..."
    pip install -q --upgrade pip
    pip install -q pytest pytest-asyncio httpx selenium python-dotenv
    
    # Install project dependencies if requirements.txt exists
    if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
        pip install -q -r "${PROJECT_ROOT}/requirements.txt"
    fi
    
    print_color $GREEN "✅ Python environment ready"
    echo ""
}

# Function to run specific test category
run_test_category() {
    local category=$1
    local test_file="test_${category}.py"
    
    print_color $BLUE "🧪 Running ${category} tests..."
    echo ""
    
    # Set PYTHONPATH to include project root
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    
    # Run the test with appropriate options
    if pytest "${PROJECT_ROOT}/tests/integration/${test_file}" \
           --tb=short \
           --verbose \
           --timeout=${TEST_TIMEOUT} \
           --capture=no \
           --color=yes; then
        print_color $GREEN "✅ ${category} tests passed"
        return 0
    else
        print_color $RED "❌ ${category} tests failed"
        return 1
    fi
}

# Function to create test database and demo users
setup_test_data() {
    print_color $BLUE "🗄️  Setting up test data..."
    
    # Create demo admin user if database creation script exists
    if [ -f "${PROJECT_ROOT}/src/database/create_admin.py" ]; then
        cd "${PROJECT_ROOT}"
        python src/database/create_admin.py || true
        print_color $GREEN "✅ Demo users ready"
    else
        print_color $YELLOW "⚠️  No admin creation script found. Manual user setup may be required."
    fi
    
    echo ""
}

# Function to run all tests
run_all_tests() {
    print_color $BLUE "🔥 Running all integration tests..."
    echo ""
    
    local passed_tests=()
    local failed_tests=()
    local total_tests=${#TEST_CATEGORIES[@]}
    
    for category in "${TEST_CATEGORIES[@]}"; do
        echo ""
        print_color $BLUE "==================== ${category^^} TESTS ===================="
        
        if run_test_category "$category"; then
            passed_tests+=("$category")
        else
            failed_tests+=("$category")
        fi
        
        echo ""
    done
    
    # Print summary
    echo ""
    print_color $BLUE "================= TEST SUMMARY =================="
    echo ""
    print_color $GREEN "✅ Passed tests (${#passed_tests[@]}/${total_tests}):"
    for test in "${passed_tests[@]}"; do
        print_color $GREEN "   - $test"
    done
    
    if [ ${#failed_tests[@]} -gt 0 ]; then
        echo ""
        print_color $RED "❌ Failed tests (${#failed_tests[@]}/${total_tests}):"
        for test in "${failed_tests[@]}"; do
            print_color $RED "   - $test"
        done
        echo ""
        print_color $RED "Some tests failed. Check the output above for details."
        return 1
    else
        echo ""
        print_color $GREEN "🎉 All tests passed! The authentication system is working correctly."
        return 0
    fi
}

# Function to run specific test
run_specific_test() {
    local test_name=$1
    
    if [[ ! " ${TEST_CATEGORIES[@]} " =~ " ${test_name} " ]]; then
        print_color $RED "❌ Unknown test category: ${test_name}"
        print_color $YELLOW "Available categories: ${TEST_CATEGORIES[*]}"
        exit 1
    fi
    
    setup_python_env
    setup_test_data
    run_test_category "$test_name"
}

# Function to show help
show_help() {
    echo "RAG Education Assistant - Integration Tests Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] [TEST_CATEGORY]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --services-only     Only check services (no tests)"
    echo "  -q, --quick             Run tests without service checks"
    echo "  --setup-only            Only setup environment and data"
    echo ""
    echo "Test Categories:"
    for category in "${TEST_CATEGORIES[@]}"; do
        echo "  ${category}"
    done
    echo ""
    echo "Examples:"
    echo "  $0                      # Run all tests"
    echo "  $0 database             # Run only database tests"
    echo "  $0 auth_system          # Run only auth system tests"
    echo "  $0 -s                   # Check services only"
    echo "  $0 --quick database     # Run database tests without service checks"
    echo ""
}

# Main execution
main() {
    cd "${PROJECT_ROOT}"
    
    local services_only=false
    local quick_mode=false
    local setup_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--services-only)
                services_only=true
                shift
                ;;
            -q|--quick)
                quick_mode=true
                shift
                ;;
            --setup-only)
                setup_only=true
                shift
                ;;
            -*)
                print_color $RED "❌ Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                # Test category
                if [ "$services_only" = true ] || [ "$setup_only" = true ]; then
                    print_color $RED "❌ Cannot specify test category with --services-only or --setup-only"
                    exit 1
                fi
                
                if [ "$quick_mode" = false ]; then
                    wait_for_services
                fi
                
                setup_python_env
                setup_test_data
                run_specific_test "$1"
                exit $?
                ;;
        esac
    done
    
    # Handle special modes
    if [ "$services_only" = true ]; then
        wait_for_services
        exit 0
    fi
    
    if [ "$setup_only" = true ]; then
        setup_python_env
        setup_test_data
        exit 0
    fi
    
    # Default: run all tests
    if [ "$quick_mode" = false ]; then
        wait_for_services
    fi
    
    setup_python_env
    setup_test_data
    
    if run_all_tests; then
        print_color $GREEN "🎉 All integration tests completed successfully!"
        exit 0
    else
        print_color $RED "💥 Some integration tests failed!"
        exit 1
    fi
}

# Trap to cleanup on exit
cleanup() {
    print_color $YELLOW "🧹 Cleaning up..."
    # Remove any temporary test files
    find "${PROJECT_ROOT}" -name "test_*.db" -delete 2>/dev/null || true
}

trap cleanup EXIT

# Run main function
main "$@"