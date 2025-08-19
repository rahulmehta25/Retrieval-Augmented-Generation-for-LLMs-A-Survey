# Staging Environment Configuration

# Environment settings
environment = "staging"
owner       = "rag-team"
aws_region  = "us-west-2"

# VPC Configuration
vpc_cidr = "10.1.0.0/16"

# EKS Configuration
kubernetes_version = "1.28"
cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]  # Restrict this based on your network

# Node Group Configuration - Production-like but smaller scale
node_instance_types = ["t3.large", "m5.large"]
compute_instance_types = ["c5.xlarge"]
node_group_min_size = 2
node_group_max_size = 8
node_group_desired_size = 3

# RDS Configuration - Production-like settings
enable_rds = true
rds_instance_class = "db.t3.small"
rds_allocated_storage = 50
rds_max_allocated_storage = 200

# High availability settings
enable_spot_instances = true  # Some cost optimization
single_nat_gateway = false   # HA across AZs
enable_nat_gateway = true

# Features - Enable all for staging validation
enable_monitoring = true
enable_logging = true
enable_irsa = true
enable_pod_security_policy = false
enable_cluster_autoscaler = true
enable_metrics_server = true
enable_aws_load_balancer_controller = true
enable_cert_manager = true
enable_backup = true

# Domain configuration
domain_name = "staging.your-domain.com"
create_route53_zone = false

# Security - Production-like security
enable_encryption_at_rest = true
enable_encryption_in_transit = true
compliance_framework = "none"

# Backup settings
backup_retention_days = 14
enable_cross_region_backup = false

# Performance settings
enable_gpu_nodes = false

# Additional tags
additional_tags = {
  CostCenter = "Engineering"
  Team = "RAG-Team"
  Purpose = "Staging and Pre-Production Testing"
  DataClassification = "Internal"
}