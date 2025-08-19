# Outputs for RAG System Infrastructure

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnets
}

output "database_subnets" {
  description = "List of database subnet IDs"
  value       = module.vpc.database_subnets
}

output "nat_gateway_ids" {
  description = "List of NAT Gateway IDs"
  value       = module.vpc.natgw_ids
}

# EKS Cluster Outputs
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks.cluster_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_version" {
  description = "The Kubernetes version of the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by EKS"
  value       = module.eks.cluster_primary_security_group_id
}

# OIDC Provider Outputs
output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster OIDC Issuer"
  value       = module.eks.cluster_oidc_issuer_url
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider for IRSA"
  value       = module.eks.oidc_provider_arn
}

# Node Group Outputs
output "eks_managed_node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

output "node_security_group_id" {
  description = "ID of the node shared security group"
  value       = module.eks.node_security_group_id
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? aws_db_instance.rag[0].endpoint : null
}

output "rds_port" {
  description = "RDS instance port"
  value       = var.enable_rds ? aws_db_instance.rag[0].port : null
}

output "rds_database_name" {
  description = "RDS database name"
  value       = var.enable_rds ? aws_db_instance.rag[0].db_name : null
}

output "rds_username" {
  description = "RDS database username"
  value       = var.enable_rds ? aws_db_instance.rag[0].username : null
  sensitive   = true
}

output "rds_secret_arn" {
  description = "ARN of the RDS password secret in AWS Secrets Manager"
  value       = var.enable_rds ? aws_secretsmanager_secret.db_password[0].arn : null
}

# S3 Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for artifacts"
  value       = aws_s3_bucket.rag_artifacts.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.rag_artifacts.arn
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = aws_s3_bucket.rag_artifacts.bucket_domain_name
}

# KMS Outputs
output "kms_key_id" {
  description = "ID of the KMS key used for encryption"
  value       = aws_kms_key.eks.key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key used for encryption"
  value       = aws_kms_key.eks.arn
}

output "kms_alias_name" {
  description = "Name of the KMS key alias"
  value       = aws_kms_alias.eks.name
}

# CloudWatch Outputs
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for application logs"
  value       = aws_cloudwatch_log_group.rag_app.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.rag_app.arn
}

# Networking Outputs
output "additional_security_group_id" {
  description = "ID of the additional security group"
  value       = aws_security_group.additional.id
}

output "rds_security_group_id" {
  description = "ID of the RDS security group"
  value       = var.enable_rds ? aws_security_group.rds[0].id : null
}

# Configuration Outputs for kubectl
output "kubectl_config" {
  description = "kubectl configuration for accessing the EKS cluster"
  value = {
    cluster_name     = module.eks.cluster_name
    cluster_endpoint = module.eks.cluster_endpoint
    cluster_ca_cert  = module.eks.cluster_certificate_authority_data
    aws_region       = var.aws_region
  }
}

# Useful commands
output "useful_commands" {
  description = "Useful commands for managing the infrastructure"
  value = {
    update_kubeconfig = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
    get_nodes         = "kubectl get nodes"
    get_pods          = "kubectl get pods -A"
    port_forward_frontend = "kubectl port-forward -n rag-${var.environment} svc/rag-frontend-service 3000:80"
    port_forward_backend  = "kubectl port-forward -n rag-${var.environment} svc/rag-backend-service 8000:8000"
    logs_backend     = "kubectl logs -n rag-${var.environment} -l app=rag-backend -f"
    logs_frontend    = "kubectl logs -n rag-${var.environment} -l app=rag-frontend -f"
  }
}

# Environment-specific outputs
output "environment_info" {
  description = "Environment-specific information"
  value = {
    environment        = var.environment
    cluster_name      = local.cluster_name
    region            = var.aws_region
    availability_zones = local.azs
    node_groups       = keys(module.eks.eks_managed_node_groups)
  }
}

# IRSA (IAM Roles for Service Accounts) helper
output "irsa_example_commands" {
  description = "Example commands for creating IRSA roles"
  value = var.enable_irsa ? {
    create_service_account = "kubectl create serviceaccount rag-backend-sa -n rag-${var.environment}"
    annotate_service_account = "kubectl annotate serviceaccount rag-backend-sa -n rag-${var.environment} eks.amazonaws.com/role-arn=arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/rag-backend-irsa-role"
  } : null
}

# Monitoring URLs (if monitoring is enabled)
output "monitoring_urls" {
  description = "URLs for monitoring services (update with actual LoadBalancer IPs after deployment)"
  value = var.enable_monitoring ? {
    grafana_url     = "http://grafana.rag-${var.environment}.your-domain.com"
    prometheus_url  = "http://prometheus.rag-${var.environment}.your-domain.com"
    alertmanager_url = "http://alertmanager.rag-${var.environment}.your-domain.com"
  } : null
}

# Application URLs
output "application_urls" {
  description = "URLs for the RAG application (update with actual domain after deployment)"
  value = {
    frontend_url = "https://rag${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"
    backend_url  = "https://api${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"
    admin_url    = "https://admin${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"
  }
}

# Cost estimation helper
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (approximate)"
  value = {
    disclaimer = "These are rough estimates and actual costs may vary significantly based on usage patterns"
    eks_cluster = "$73 (control plane) + node costs"
    node_groups = "Variable based on instance types and scaling"
    rds = var.enable_rds ? "~$15-200+ depending on instance class" : "Not enabled"
    nat_gateway = var.enable_nat_gateway ? (var.single_nat_gateway ? "~$32" : "~$96 (3 AZs)") : "Not enabled"
    load_balancer = "~$18 per ALB/NLB"
    data_transfer = "Variable based on traffic"
    storage = "Variable based on EBS volumes and S3 usage"
  }
}

# Security information
output "security_info" {
  description = "Security configuration summary"
  value = {
    encryption_at_rest = var.enable_encryption_at_rest
    encryption_in_transit = var.enable_encryption_in_transit
    kms_key_arn = aws_kms_key.eks.arn
    vpc_flow_logs_enabled = true
    private_subnets_only = "EKS nodes are in private subnets"
    security_groups = "Restrictive security groups configured"
    irsa_enabled = var.enable_irsa
    pod_security_policy = var.enable_pod_security_policy
  }
}

# Backup information
output "backup_info" {
  description = "Backup configuration summary"
  value = var.enable_backup ? {
    rds_backup_retention = var.enable_rds ? "7-30 days based on environment" : "N/A"
    s3_versioning = "Enabled on artifacts bucket"
    cross_region_backup = var.enable_cross_region_backup ? "Enabled to ${var.backup_region}" : "Disabled"
    automated_snapshots = "Configured via Kubernetes CronJobs"
  } : "Backup is disabled"
}

# Troubleshooting information
output "troubleshooting" {
  description = "Common troubleshooting commands and information"
  value = {
    check_cluster_status = "aws eks describe-cluster --region ${var.aws_region} --name ${module.eks.cluster_name}"
    check_node_groups = "aws eks describe-nodegroup --region ${var.aws_region} --cluster-name ${module.eks.cluster_name} --nodegroup-name general"
    check_addons = "aws eks list-addons --region ${var.aws_region} --cluster-name ${module.eks.cluster_name}"
    cluster_autoscaler_logs = "kubectl logs -n kube-system -l app=cluster-autoscaler"
    aws_load_balancer_controller_logs = "kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller"
  }
}