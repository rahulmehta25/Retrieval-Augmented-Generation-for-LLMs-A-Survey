# Variables for RAG System Infrastructure

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rag-system"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "rag-team"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the EKS cluster endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Node Group Configuration
variable "node_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "compute_instance_types" {
  description = "Instance types for compute-optimized node group"
  type        = list(string)
  default     = ["c5.xlarge", "c5.2xlarge"]
}

variable "node_group_min_size" {
  description = "Minimum size of the node group"
  type        = number
  default     = 1
}

variable "node_group_max_size" {
  description = "Maximum size of the node group"
  type        = number
  default     = 10
}

variable "node_group_desired_size" {
  description = "Desired size of the node group"
  type        = number
  default     = 3
}

# RDS Configuration
variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = true
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
  validation {
    condition = contains([
      "db.t3.micro", "db.t3.small", "db.t3.medium", "db.t3.large",
      "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge", "db.r5.4xlarge"
    ], var.rds_instance_class)
    error_message = "RDS instance class must be a valid PostgreSQL instance type."
  }
}

variable "rds_allocated_storage" {
  description = "Initial allocated storage for RDS (GB)"
  type        = number
  default     = 20
  validation {
    condition     = var.rds_allocated_storage >= 20 && var.rds_allocated_storage <= 65536
    error_message = "RDS allocated storage must be between 20 and 65536 GB."
  }
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage for RDS auto-scaling (GB)"
  type        = number
  default     = 100
}

# Monitoring and Logging
variable "enable_monitoring" {
  description = "Enable comprehensive monitoring stack"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging"
  type        = bool
  default     = true
}

# Security
variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = true
}

# Networking
variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use a single NAT Gateway for all private subnets (cost optimization for dev)"
  type        = bool
  default     = false
}

# Storage
variable "ebs_csi_driver_version" {
  description = "Version of the EBS CSI driver"
  type        = string
  default     = "v1.24.0-eksbuild.1"
}

# Auto Scaling
variable "enable_cluster_autoscaler" {
  description = "Enable Cluster Autoscaler"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable Metrics Server for HPA"
  type        = bool
  default     = true
}

# Load Balancing
variable "enable_aws_load_balancer_controller" {
  description = "Enable AWS Load Balancer Controller"
  type        = bool
  default     = true
}

# DNS
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "your-domain.com"
}

variable "create_route53_zone" {
  description = "Create Route53 hosted zone"
  type        = bool
  default     = false
}

# Certificates
variable "enable_cert_manager" {
  description = "Enable cert-manager for automatic SSL certificates"
  type        = bool
  default     = true
}

# Backup
variable "enable_backup" {
  description = "Enable backup solutions"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention days must be between 1 and 365."
  }
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["t3.medium", "t3.large", "m5.large", "m5.xlarge"]
}

# Environment-specific defaults
locals {
  environment_defaults = {
    dev = {
      node_group_min_size      = 1
      node_group_max_size      = 5
      node_group_desired_size  = 2
      rds_instance_class      = "db.t3.micro"
      rds_allocated_storage   = 20
      single_nat_gateway      = true
      enable_spot_instances   = true
      backup_retention_days   = 7
    }
    staging = {
      node_group_min_size      = 2
      node_group_max_size      = 8
      node_group_desired_size  = 3
      rds_instance_class      = "db.t3.small"
      rds_allocated_storage   = 50
      single_nat_gateway      = false
      enable_spot_instances   = true
      backup_retention_days   = 14
    }
    prod = {
      node_group_min_size      = 3
      node_group_max_size      = 20
      node_group_desired_size  = 5
      rds_instance_class      = "db.r5.large"
      rds_allocated_storage   = 100
      single_nat_gateway      = false
      enable_spot_instances   = false
      backup_retention_days   = 30
    }
  }
}

# Additional configuration variables
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "kubernetes_labels" {
  description = "Additional Kubernetes labels"
  type        = map(string)
  default     = {}
}

variable "kubernetes_annotations" {
  description = "Additional Kubernetes annotations"
  type        = map(string)
  default     = {}
}

# Secrets and credentials
variable "github_token" {
  description = "GitHub token for CI/CD integration"
  type        = string
  default     = ""
  sensitive   = true
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  default     = ""
  sensitive   = true
}

# Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Region for cross-region backup replication"
  type        = string
  default     = "us-east-1"
}

# Performance
variable "enable_gpu_nodes" {
  description = "Enable GPU-enabled nodes for ML workloads"
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "GPU instance types"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge"]
}

# Compliance
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all storage"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "compliance_framework" {
  description = "Compliance framework (hipaa, pci, soc2, none)"
  type        = string
  default     = "none"
  validation {
    condition     = contains(["none", "hipaa", "pci", "soc2"], var.compliance_framework)
    error_message = "Compliance framework must be one of: none, hipaa, pci, soc2."
  }
}