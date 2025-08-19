#!/bin/bash

# RAG System Deployment Script
# This script handles the complete deployment of the RAG system to different environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"
K8S_DIR="$PROJECT_ROOT/k8s"

# Default values
ENVIRONMENT=""
ACTION="plan"
AUTO_APPROVE=false
DESTROY=false
SKIP_TERRAFORM=false
SKIP_KUBERNETES=false
DRY_RUN=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the RAG system infrastructure and applications.

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, prod)
    -a, --action ACTION      Action to perform (plan, apply, destroy)
    -y, --auto-approve       Auto-approve Terraform changes
    -d, --destroy            Destroy infrastructure (same as --action destroy)
    --skip-terraform         Skip Terraform deployment
    --skip-kubernetes        Skip Kubernetes deployment
    --dry-run               Show what would be done without executing
    -h, --help              Show this help message

EXAMPLES:
    $0 -e dev -a plan                    # Plan dev environment
    $0 -e staging -a apply -y            # Apply staging with auto-approve
    $0 -e prod -a apply                  # Apply production (interactive)
    $0 -e dev -d -y                      # Destroy dev environment
    $0 -e prod --skip-terraform          # Deploy only Kubernetes to prod

ENVIRONMENTS:
    dev         - Development environment with minimal resources
    staging     - Staging environment with production-like setup
    prod        - Production environment with high availability

ACTIONS:
    plan        - Show what Terraform will do
    apply       - Create/update infrastructure and deploy applications
    destroy     - Destroy all infrastructure (use with caution!)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -y|--auto-approve)
            AUTO_APPROVE=true
            shift
            ;;
        -d|--destroy)
            DESTROY=true
            ACTION="destroy"
            shift
            ;;
        --skip-terraform)
            SKIP_TERRAFORM=true
            shift
            ;;
        --skip-kubernetes)
            SKIP_KUBERNETES=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$ENVIRONMENT" ]]; then
    print_error "Environment is required. Use -e or --environment"
    usage
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod"
    exit 1
fi

# Validate action
if [[ ! "$ACTION" =~ ^(plan|apply|destroy)$ ]]; then
    print_error "Invalid action: $ACTION. Must be plan, apply, or destroy"
    exit 1
fi

# Safety check for production
if [[ "$ENVIRONMENT" == "prod" && "$ACTION" == "destroy" ]]; then
    if [[ "$AUTO_APPROVE" == "true" ]]; then
        print_error "Auto-approve is not allowed when destroying production!"
        exit 1
    fi
    
    print_warning "You are about to DESTROY the PRODUCTION environment!"
    print_warning "This will delete ALL production data and infrastructure!"
    echo
    read -p "Type 'DESTROY PRODUCTION' to continue: " confirmation
    
    if [[ "$confirmation" != "DESTROY PRODUCTION" ]]; then
        print_error "Confirmation failed. Aborting."
        exit 1
    fi
fi

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v terraform >/dev/null 2>&1 || missing_tools+=("terraform")
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v aws >/dev/null 2>&1 || missing_tools+=("aws")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        print_error "AWS credentials not configured or expired"
        exit 1
    fi
    
    # Check Terraform files exist
    if [[ ! -f "$TERRAFORM_DIR/main.tf" ]]; then
        print_error "Terraform files not found in $TERRAFORM_DIR"
        exit 1
    fi
    
    # Check environment-specific tfvars file
    if [[ ! -f "$TERRAFORM_DIR/environments/${ENVIRONMENT}.tfvars" ]]; then
        print_error "Environment file not found: $TERRAFORM_DIR/environments/${ENVIRONMENT}.tfvars"
        exit 1
    fi
    
    print_success "Prerequisites check completed"
}

# Function to setup Terraform backend
setup_terraform_backend() {
    print_status "Setting up Terraform backend..."
    
    local backend_bucket="rag-terraform-state-${ENVIRONMENT}-$(date +%s)"
    local backend_region="${AWS_DEFAULT_REGION:-us-west-2}"
    local dynamodb_table="terraform-locks-${ENVIRONMENT}"
    
    # Create S3 bucket for state if it doesn't exist
    if [[ "$DRY_RUN" == "false" ]]; then
        aws s3api head-bucket --bucket "$backend_bucket" 2>/dev/null || {
            print_status "Creating Terraform state bucket: $backend_bucket"
            aws s3api create-bucket \
                --bucket "$backend_bucket" \
                --region "$backend_region" \
                --create-bucket-configuration LocationConstraint="$backend_region"
            
            # Enable versioning
            aws s3api put-bucket-versioning \
                --bucket "$backend_bucket" \
                --versioning-configuration Status=Enabled
            
            # Enable encryption
            aws s3api put-bucket-encryption \
                --bucket "$backend_bucket" \
                --server-side-encryption-configuration '{
                    "Rules": [{
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }]
                }'
        }
        
        # Create DynamoDB table for locking if it doesn't exist
        aws dynamodb describe-table --table-name "$dynamodb_table" 2>/dev/null || {
            print_status "Creating DynamoDB table for state locking: $dynamodb_table"
            aws dynamodb create-table \
                --table-name "$dynamodb_table" \
                --attribute-definitions AttributeName=LockID,AttributeType=S \
                --key-schema AttributeName=LockID,KeyType=HASH \
                --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
        }
    fi
    
    # Create backend configuration
    cat > "$TERRAFORM_DIR/backend-${ENVIRONMENT}.hcl" << EOF
bucket         = "$backend_bucket"
key            = "rag-system/${ENVIRONMENT}/terraform.tfstate"
region         = "$backend_region"
encrypt        = true
dynamodb_table = "$dynamodb_table"
EOF
    
    print_success "Terraform backend setup completed"
}

# Function to deploy infrastructure with Terraform
deploy_terraform() {
    if [[ "$SKIP_TERRAFORM" == "true" ]]; then
        print_warning "Skipping Terraform deployment"
        return 0
    fi
    
    print_status "Deploying infrastructure with Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    print_status "Initializing Terraform..."
    if [[ "$DRY_RUN" == "false" ]]; then
        terraform init -backend-config="backend-${ENVIRONMENT}.hcl" -upgrade
    else
        print_status "[DRY-RUN] Would run: terraform init -backend-config=backend-${ENVIRONMENT}.hcl -upgrade"
    fi
    
    # Validate configuration
    print_status "Validating Terraform configuration..."
    if [[ "$DRY_RUN" == "false" ]]; then
        terraform validate
    else
        print_status "[DRY-RUN] Would run: terraform validate"
    fi
    
    # Format check
    terraform fmt -check=true -diff=true || {
        print_warning "Terraform files are not properly formatted. Run 'terraform fmt' to fix."
    }
    
    # Plan or Apply
    local tf_args=("-var-file=environments/${ENVIRONMENT}.tfvars")
    
    if [[ "$AUTO_APPROVE" == "true" ]]; then
        tf_args+=("-auto-approve")
    fi
    
    if [[ "$ACTION" == "plan" ]]; then
        print_status "Running Terraform plan..."
        if [[ "$DRY_RUN" == "false" ]]; then
            terraform plan "${tf_args[@]}" -out="tfplan-${ENVIRONMENT}"
        else
            print_status "[DRY-RUN] Would run: terraform plan ${tf_args[*]} -out=tfplan-${ENVIRONMENT}"
        fi
    elif [[ "$ACTION" == "apply" ]]; then
        print_status "Applying Terraform changes..."
        if [[ "$DRY_RUN" == "false" ]]; then
            terraform apply "${tf_args[@]}"
            
            # Save outputs
            terraform output -json > "outputs-${ENVIRONMENT}.json"
        else
            print_status "[DRY-RUN] Would run: terraform apply ${tf_args[*]}"
        fi
    elif [[ "$ACTION" == "destroy" ]]; then
        print_status "Destroying infrastructure..."
        if [[ "$DRY_RUN" == "false" ]]; then
            terraform destroy "${tf_args[@]}"
        else
            print_status "[DRY-RUN] Would run: terraform destroy ${tf_args[*]}"
        fi
    fi
    
    cd "$PROJECT_ROOT"
    print_success "Terraform deployment completed"
}

# Function to setup kubectl configuration
setup_kubectl() {
    if [[ "$ACTION" == "destroy" || "$SKIP_KUBERNETES" == "true" ]]; then
        return 0
    fi
    
    print_status "Setting up kubectl configuration..."
    
    # Get cluster name from Terraform outputs
    local cluster_name
    if [[ -f "$TERRAFORM_DIR/outputs-${ENVIRONMENT}.json" ]]; then
        cluster_name=$(jq -r '.cluster_name.value' "$TERRAFORM_DIR/outputs-${ENVIRONMENT}.json")
    else
        cluster_name="rag-system-${ENVIRONMENT}-cluster"
        print_warning "Using default cluster name: $cluster_name"
    fi
    
    # Update kubeconfig
    if [[ "$DRY_RUN" == "false" ]]; then
        aws eks update-kubeconfig --region "${AWS_DEFAULT_REGION:-us-west-2}" --name "$cluster_name"
        
        # Test connection
        kubectl get nodes
    else
        print_status "[DRY-RUN] Would run: aws eks update-kubeconfig --region ${AWS_DEFAULT_REGION:-us-west-2} --name $cluster_name"
    fi
    
    print_success "kubectl configuration completed"
}

# Function to deploy Kubernetes applications
deploy_kubernetes() {
    if [[ "$ACTION" == "destroy" || "$SKIP_KUBERNETES" == "true" ]]; then
        return 0
    fi
    
    print_status "Deploying Kubernetes applications..."
    
    # Deploy using Kustomize
    local overlay_path="$K8S_DIR/overlays/$ENVIRONMENT"
    
    if [[ ! -d "$overlay_path" ]]; then
        print_error "Kubernetes overlay not found: $overlay_path"
        return 1
    fi
    
    # Replace image tags with latest built images
    if [[ -f "$PROJECT_ROOT/.env.${ENVIRONMENT}" ]]; then
        source "$PROJECT_ROOT/.env.${ENVIRONMENT}"
        
        if [[ -n "${BACKEND_IMAGE_TAG:-}" ]]; then
            sed -i.bak "s/BACKEND_IMAGE_TAG/${BACKEND_IMAGE_TAG}/g" "$overlay_path/kustomization.yml"
        fi
        
        if [[ -n "${FRONTEND_IMAGE_TAG:-}" ]]; then
            sed -i.bak "s/FRONTEND_IMAGE_TAG/${FRONTEND_IMAGE_TAG}/g" "$overlay_path/kustomization.yml"
        fi
    fi
    
    # Apply Kubernetes manifests
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl apply -k "$overlay_path"
        
        # Wait for deployments to be ready
        print_status "Waiting for deployments to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment --all -n "rag-${ENVIRONMENT}"
        
        # Get service URLs
        print_status "Getting service information..."
        kubectl get services -n "rag-${ENVIRONMENT}"
    else
        print_status "[DRY-RUN] Would run: kubectl apply -k $overlay_path"
    fi
    
    print_success "Kubernetes deployment completed"
}

# Function to run post-deployment checks
post_deployment_checks() {
    if [[ "$ACTION" == "destroy" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    print_status "Running post-deployment checks..."
    
    # Check pod status
    print_status "Checking pod status..."
    kubectl get pods -n "rag-${ENVIRONMENT}" --field-selector=status.phase!=Running || true
    
    # Check service endpoints
    print_status "Checking service endpoints..."
    kubectl get endpoints -n "rag-${ENVIRONMENT}"
    
    # Basic health checks
    if command -v curl >/dev/null 2>&1; then
        local backend_url
        backend_url=$(kubectl get svc -n "rag-${ENVIRONMENT}" rag-backend-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "localhost")
        
        if [[ "$backend_url" != "localhost" ]]; then
            print_status "Testing backend health endpoint..."
            curl -f "http://${backend_url}:8000/health" || print_warning "Backend health check failed"
        fi
    fi
    
    print_success "Post-deployment checks completed"
}

# Function to cleanup on exit
cleanup() {
    if [[ -f "$TERRAFORM_DIR/backend-${ENVIRONMENT}.hcl" ]]; then
        rm -f "$TERRAFORM_DIR/backend-${ENVIRONMENT}.hcl"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    print_status "Starting RAG system deployment for environment: $ENVIRONMENT"
    print_status "Action: $ACTION"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "DRY-RUN MODE: No actual changes will be made"
    fi
    
    check_prerequisites
    setup_terraform_backend
    deploy_terraform
    
    if [[ "$ACTION" != "destroy" ]]; then
        setup_kubectl
        deploy_kubernetes
        post_deployment_checks
        
        print_success "Deployment completed successfully!"
        print_status "You can access your applications using the service URLs displayed above."
        print_status "For monitoring, check: kubectl get all -n rag-${ENVIRONMENT}"
    else
        print_success "Infrastructure destroyed successfully!"
    fi
}

# Run main function
main