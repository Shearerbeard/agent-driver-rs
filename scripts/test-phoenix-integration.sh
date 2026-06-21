#!/bin/bash
# Phoenix Integration Test Runner
#
# Usage:
#   ./scripts/test-phoenix-integration.sh          # Run all tests
#   ./scripts/test-phoenix-integration.sh mock      # Run only mock tests
#   ./scripts/test-phoenix-integration.sh bedrock   # Run Bedrock tests (requires AWS credentials)
#   ./scripts/test-phoenix-integration.sh ollama    # Run Ollama tests (requires Ollama running)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Phoenix Integration Test Suite${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Default: run all tests
TEST_TYPE="${1:-all}"

# Function to start Phoenix
start_phoenix() {
    echo -e "${YELLOW}📦 Starting Phoenix...${NC}"
    docker compose -f "$PROJECT_ROOT/docker-compose.phoenix.yaml" up -d phoenix

    echo -e "${YELLOW}⏳ Waiting for Phoenix to be ready...${NC}"
    for i in {1..30}; do
        if curl -sf http://localhost:6006/healthz > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Phoenix is ready${NC}"
            return 0
        fi
        sleep 1
    done
    echo -e "${RED}❌ Phoenix failed to start${NC}"
    return 1
}

# Function to stop Phoenix
stop_phoenix() {
    echo -e "${YELLOW}🛑 Stopping Phoenix...${NC}"
    docker compose -f "$PROJECT_ROOT/docker-compose.phoenix.yaml" down
}

# Function to show Phoenix logs
show_logs() {
    echo ""
    echo -e "${YELLOW}📜 Phoenix logs (last 50 lines):${NC}"
    echo "──────────────────────────────────────────────────────"
    docker compose -f "$PROJECT_ROOT/docker-compose.phoenix.yaml" logs phoenix | tail -50
    echo "──────────────────────────────────────────────────────"
}

# Run mock tests
run_mock_tests() {
    echo -e "${BLUE}🧪 Running Mock Provider Tests${NC}"
    echo "──────────────────────────────────────────────────────"

    cargo run --example phoenix_integration_mock --features phoenix 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Mock tests passed${NC}"
    else
        echo -e "${RED}❌ Mock tests failed${NC}"
        return 1
    fi
}

# Run Bedrock tests
run_bedrock_tests() {
    echo -e "${BLUE}🧪 Running Bedrock Provider Tests${NC}"
    echo "──────────────────────────────────────────────────────"
    echo -e "${YELLOW}⚠️  Bedrock tests require AWS credentials${NC}"
    echo -e "${YELLOW}⚠️  Make sure you're logged in via SSO or have AWS credentials set${NC}"
    echo ""

    # Check for AWS credentials
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        echo -e "${RED}❌ AWS credentials not found. Run 'aws sso login' first.${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ AWS credentials found${NC}"

    # Run Bedrock test example
    echo -e "${YELLOW}Running Bedrock integration test...${NC}"
    cargo run --example phoenix_integration_bedrock --features "phoenix bedrock" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Bedrock tests passed${NC}"
    else
        echo -e "${RED}❌ Bedrock tests failed${NC}"
        return 1
    fi
}

# Run Ollama tests
run_ollama_tests() {
    echo -e "${BLUE}🧪 Running Ollama Provider Tests${NC}"
    echo "──────────────────────────────────────────────────────"
    echo -e "${YELLOW}⚠️  Ollama tests require Ollama to be running${NC}"
    echo ""

    # Check for Ollama
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${RED}❌ Ollama is not running. Start it with 'ollama serve'${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Ollama is running${NC}"

    # Run Ollama test example
    echo -e "${Yellow}Running Ollama integration test...${NC}"
    cargo run --example phoenix_integration_ollama --features "phoenix ollama" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Ollama tests passed${NC}"
    else
        echo -e "${RED}❌ Ollama tests failed${NC}"
        return 1
    fi
}

# Cleanup on exit (only if phoenix was started)
cleanup() {
    if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "bedrock" ] || [ "$TEST_TYPE" = "ollama" ]; then
        PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
        stop_phoenix
    fi
}
trap cleanup EXIT

# Main logic
case "$TEST_TYPE" in
    mock)
        run_mock_tests
        ;;
    bedrock)
        run_bedrock_tests
        ;;
    ollama)
        run_ollama_tests
        ;;
    all)
        start_phoenix
        run_mock_tests
        show_logs
        ;;
    start)
        start_phoenix
        echo -e "${GREEN}✅ Phoenix is running${NC}"
        echo "  - UI:    http://localhost:6006"
        echo "  - OTLP:  localhost:4317"
        echo ""
        echo "View logs: docker compose -f $PROJECT_ROOT/docker-compose.phoenix.yaml logs -f phoenix"
        ;;
    stop)
        stop_phoenix
        echo -e "${GREEN}✅ Phoenix stopped${NC}"
        ;;
    logs)
        docker compose -f "$PROJECT_ROOT/docker-compose.phoenix.yaml" logs -f phoenix
        ;;
    *)
        echo "Usage: $0 [mock|bedrock|ollama|all|start|stop|logs]"
        echo ""
        echo "  mock    - Run mock provider tests (default)"
        echo "  bedrock - Run Bedrock provider tests (requires AWS)"
        echo "  ollama  - Run Ollama provider tests (requires Ollama)"
        echo "  all     - Run all tests with Phoenix"
        echo "  start   - Start Phoenix only"
        echo "  stop    - Stop Phoenix only"
        echo "  logs    - View Phoenix logs"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Done!${NC}"
