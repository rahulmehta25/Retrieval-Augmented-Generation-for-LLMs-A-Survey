# Development Environment Configuration

# Environment settings
environment = "dev"
owner       = "dev-team"
aws_region  = "us-west-2"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"

# EKS Configuration
kubernetes_version = "1.28"
cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]  # Restrict this in production

# Node Group Configuration - Cost optimized for development
node_instance_types = ["t3.medium", "t3.large"]
compute_instance_types = ["c5.large"]
node_group_min_size = 1
node_group_max_size = 5
node_group_desired_size = 2

# RDS Configuration - Minimal for development
enable_rds = true
rds_instance_class = "db.t3.micro"
rds_allocated_storage = 20
rds_max_allocated_storage = 50

# Cost optimization settings
enable_spot_instances = true
single_nat_gateway = true  # Cost optimization
enable_nat_gateway = true

# Features - Enable most for development testing
enable_monitoring = true
enable_logging = true
enable_irsa = true
enable_pod_security_policy = false  # Deprecated in newer K8s versions
enable_cluster_autoscaler = true
enable_metrics_server = true
enable_aws_load_balancer_controller = true
enable_cert_manager = true
enable_backup = false  # Disable backup in dev to save costs

# Domain configuration
domain_name = "dev.your-domain.com"
create_route53_zone = false

# Security - Relaxed for development
enable_encryption_at_rest = true
enable_encryption_in_transit = true
compliance_framework = "none"

# Backup settings
backup_retention_days = 7
enable_cross_region_backup = false

# Performance settings
enable_gpu_nodes = false
enable_cross_region_backup = false

# Additional tags
additional_tags = {
  CostCenter = "Development"
  Team = "RAG-Dev-Team"
  Purpose = "Development and Testing"
}