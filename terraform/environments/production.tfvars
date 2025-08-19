# Production Environment Configuration

# Environment settings
environment = "prod"
owner       = "rag-platform-team"
aws_region  = "us-west-2"

# VPC Configuration
vpc_cidr = "10.2.0.0/16"

# EKS Configuration
kubernetes_version = "1.28"
# Restrict access to production cluster - replace with your organization's IPs
cluster_endpoint_public_access_cidrs = [
  "203.0.113.0/24",  # Replace with your office IP ranges
  "198.51.100.0/24"  # Replace with your VPN IP ranges
]

# Node Group Configuration - Production scale and reliability
node_instance_types = ["m5.large", "m5.xlarge"]
compute_instance_types = ["c5.xlarge", "c5.2xlarge"]
node_group_min_size = 3
node_group_max_size = 20
node_group_desired_size = 5

# RDS Configuration - Production specifications
enable_rds = true
rds_instance_class = "db.r5.large"
rds_allocated_storage = 100
rds_max_allocated_storage = 1000

# High availability and reliability settings
enable_spot_instances = false  # Use on-demand for production reliability
single_nat_gateway = false     # HA across multiple AZs
enable_nat_gateway = true

# Features - All enabled for production
enable_monitoring = true
enable_logging = true
enable_irsa = true
enable_pod_security_policy = false  # Use Pod Security Standards instead
enable_cluster_autoscaler = true
enable_metrics_server = true
enable_aws_load_balancer_controller = true
enable_cert_manager = true
enable_backup = true

# Domain configuration
domain_name = "your-domain.com"
create_route53_zone = false  # Assuming you manage DNS separately

# Security - Maximum security for production
enable_encryption_at_rest = true
enable_encryption_in_transit = true
compliance_framework = "soc2"  # Adjust based on your compliance requirements

# Backup settings - Comprehensive backup strategy
backup_retention_days = 30
enable_cross_region_backup = true
backup_region = "us-east-1"

# Performance settings
enable_gpu_nodes = false  # Enable if you need GPU acceleration for ML workloads

# Additional production-specific tags
additional_tags = {
  CostCenter = "Production"
  Team = "RAG-Platform-Team"
  Purpose = "Production RAG System"
  DataClassification = "Confidential"
  BackupRequired = "true"
  MonitoringRequired = "true"
  ComplianceFramework = "SOC2"
  BusinessCriticality = "High"
  MaintenanceWindow = "Sunday-02:00-06:00-UTC"
  SupportTier = "24x7"
}