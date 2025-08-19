# RAG System - Comprehensive CI/CD Pipeline & Deployment Guide

This document provides a complete guide to deploying the RAG system using the comprehensive CI/CD pipeline that has been set up.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Components](#infrastructure-components)
4. [Environment Setup](#environment-setup)
5. [Deployment Process](#deployment-process)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)
10. [Cost Optimization](#cost-optimization)

## 🏗️ Architecture Overview

The RAG system is deployed with a production-ready architecture featuring:

- **Multi-environment support**: Development, Staging, and Production
- **Blue-green deployments** for zero-downtime updates
- **Auto-scaling** based on CPU, memory, and custom metrics
- **Comprehensive monitoring** with Prometheus, Grafana, and Loki
- **Infrastructure as Code** using Terraform
- **Container orchestration** with Kubernetes on AWS EKS
- **Automated CI/CD** with GitHub Actions

### Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Vector DB     │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (ChromaDB)    │
│   Nginx         │    │   Python        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────┐
         │           AWS EKS Cluster               │
         │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
         │  │  Node   │ │  Node   │ │  Node   │   │
         │  │ Group 1 │ │ Group 2 │ │ Group 3 │   │
         │  └─────────┘ └─────────┘ └─────────┘   │
         └─────────────────────────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌───▼────┐              ┌───────▼────┐                ┌──────▼──────┐
│   S3   │              │    RDS     │                │ Monitoring  │
│Backups │              │PostgreSQL  │                │Prometheus   │
│        │              │            │                │  Grafana    │
└────────┘              └────────────┘                └─────────────┘
```

## 🛠️ Prerequisites

### Required Tools

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install Terraform
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### AWS Setup

1. **Configure AWS Credentials**:
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region
   ```

2. **Create IAM User with Required Permissions**:
   - `AmazonEKSClusterPolicy`
   - `AmazonEKSWorkerNodePolicy`
   - `AmazonEKS_CNI_Policy`
   - `AmazonEC2ContainerRegistryReadOnly`
   - `AmazonS3FullAccess`
   - `AmazonRDSFullAccess`
   - `IAMFullAccess`
   - `AmazonVPCFullAccess`

3. **Domain Setup** (Optional):
   ```bash
   # If using custom domain, create Route53 hosted zone
   aws route53 create-hosted-zone --name your-domain.com --caller-reference $(date +%s)
   ```

## 🏗️ Infrastructure Components

### AWS Resources Created

#### Core Infrastructure
- **VPC** with public/private subnets across 3 AZs
- **EKS Cluster** with managed node groups
- **Application Load Balancer** with SSL termination
- **NAT Gateways** for private subnet internet access
- **Security Groups** with least privilege access

#### Storage
- **RDS PostgreSQL** for application metadata
- **S3 Buckets** for backups and artifacts
- **EBS Volumes** for persistent storage

#### Security
- **KMS Keys** for encryption at rest
- **IAM Roles** for service accounts (IRSA)
- **VPC Flow Logs** for network monitoring

#### Monitoring
- **CloudWatch** for AWS resource monitoring
- **Prometheus** for application metrics
- **Grafana** for visualization
- **Loki** for log aggregation

## 🌍 Environment Setup

### 1. Development Environment

```bash
# Deploy development environment
./scripts/deploy.sh --environment dev --action apply --auto-approve

# Access development services
kubectl port-forward -n rag-dev svc/rag-frontend-service 3000:80
kubectl port-forward -n rag-dev svc/rag-backend-service 8000:8000
```

**Features:**
- Minimal resource allocation for cost optimization
- Single NAT Gateway
- Spot instances enabled
- Basic monitoring
- 7-day backup retention

### 2. Staging Environment

```bash
# Deploy staging environment
./scripts/deploy.sh --environment staging --action apply

# Access staging services via ingress
curl https://rag-staging.your-domain.com
curl https://api-staging.rag.your-domain.com/health
```

**Features:**
- Production-like configuration at smaller scale
- Blue-green deployment support
- Full monitoring stack
- 14-day backup retention
- SSL certificates via cert-manager

### 3. Production Environment

```bash
# Deploy production environment (requires manual approval)
./scripts/deploy.sh --environment prod --action apply

# Access production services
curl https://rag.your-domain.com
curl https://api.rag.your-domain.com/health
```

**Features:**
- High availability across multiple AZs
- Auto-scaling up to 50 pods
- Cross-region backup
- 30-day backup retention
- Advanced security policies
- Performance optimized

## 🚀 Deployment Process

### Automated Deployment via GitHub Actions

1. **Trigger Deployment**:
   ```bash
   # Push to develop branch triggers dev deployment
   git push origin develop
   
   # Push to main branch triggers staging, then production
   git push origin main
   
   # Create release tag for production deployment
   git tag -a v1.0.0 -m "Production release v1.0.0"
   git push origin v1.0.0
   ```

2. **Manual Deployment**:
   ```bash
   # Plan deployment
   ./scripts/deploy.sh -e prod -a plan
   
   # Apply with confirmation
   ./scripts/deploy.sh -e prod -a apply
   
   # Destroy environment (use with caution!)
   ./scripts/deploy.sh -e dev -d -y
   ```

### Blue-Green Deployment Process

1. **Deploy to Green Environment**:
   ```bash
   # GitHub Actions automatically handles this
   # Or manually:
   kubectl apply -k k8s/overlays/production/
   ```

2. **Run Health Checks**:
   ```bash
   # Automated health checks run as part of pipeline
   kubectl get pods -n rag-prod -l color=green
   curl -f https://api.rag.your-domain.com/health
   ```

3. **Switch Traffic**:
   ```bash
   # Automated traffic switch after successful health checks
   kubectl patch svc rag-app-active -n rag-prod -p '{"spec":{"selector":{"color":"green"}}}'
   ```

4. **Rollback if Needed**:
   ```bash
   # Automatic rollback on failure
   kubectl patch svc rag-app-active -n rag-prod -p '{"spec":{"selector":{"color":"blue"}}}'
   ```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Continuous Integration (`ci.yml`)
- **Triggers**: Push to any branch, PRs
- **Jobs**:
  - Backend CI (Python testing, linting, type checking)
  - Frontend CI (React build, ESLint, TypeScript)
  - Security scanning (Trivy, CodeQL, Bandit)
  - Integration tests
  - Build validation

#### 2. Development Deployment (`cd-dev.yml`)
- **Triggers**: Push to `develop` branch
- **Jobs**:
  - Build and push Docker images
  - Deploy to development environment
  - Run smoke tests
  - Send notifications

#### 3. Production Deployment (`cd-prod.yml`)
- **Triggers**: Push to `main` branch, release tags
- **Jobs**:
  - Build production-optimized images
  - Deploy to staging with blue-green
  - Run comprehensive tests
  - Deploy to production with gradual rollout
  - Monitor and rollback if needed

### Required GitHub Secrets

```bash
# AWS Access
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Container Registry
GITHUB_TOKEN=your-github-token

# Kubernetes
KUBE_CONFIG_DEV=base64-encoded-kubeconfig-dev
KUBE_CONFIG_STAGING=base64-encoded-kubeconfig-staging  
KUBE_CONFIG_PROD=base64-encoded-kubeconfig-prod

# Notifications
SLACK_WEBHOOK_URL=your-slack-webhook
SLACK_WEBHOOK_PROD=your-production-slack-webhook
EMAIL_USERNAME=your-email
EMAIL_PASSWORD=your-app-password
NOTIFICATION_EMAIL=alerts@your-domain.com

# API Keys (if using external services)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## 📊 Monitoring & Observability

### Metrics (Prometheus + Grafana)

Access Grafana:
```bash
# Port forward to Grafana
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80

# Or via ingress
open https://grafana.your-domain.com
# Username: admin
# Password: Retrieved from AWS Secrets Manager
```

**Available Dashboards:**
- RAG System Overview
- Kubernetes Cluster Metrics
- Application Performance Monitoring
- Infrastructure Utilization
- Error Rate and Latency Tracking

### Logging (Loki + Promtail)

```bash
# View logs in Grafana
# Or via kubectl
kubectl logs -n rag-prod -l app=rag-backend -f

# Query logs with LogQL in Grafana
{namespace="rag-prod", app="rag-backend"} |= "ERROR"
```

### Alerting (AlertManager)

**Configured Alerts:**
- High error rate (>5% for 2 minutes)
- High latency (95th percentile >2s for 5 minutes)
- Pod restarts (>5 restarts in 1 hour)
- High memory usage (>90% for 5 minutes)
- High CPU usage (>90% for 10 minutes)
- Database connectivity issues
- Storage space running low

### Health Checks

```bash
# Application health
curl https://api.rag.your-domain.com/health

# Kubernetes health
kubectl get componentstatuses
kubectl get nodes
kubectl get pods -A --field-selector=status.phase!=Running

# Infrastructure health (via monitoring)
curl https://prometheus.your-domain.com/api/v1/query?query=up
```

## 🔒 Security

### Network Security
- **VPC** with private subnets for application workloads
- **Security Groups** with minimal required ports
- **Network Policies** to restrict pod-to-pod communication
- **WAF** integration for web application firewall

### Data Security
- **Encryption at rest** using KMS keys
- **Encryption in transit** with TLS 1.3
- **Secrets management** via AWS Secrets Manager
- **RBAC** for Kubernetes access control

### Container Security
- **Non-root containers** with read-only root filesystems
- **Pod Security Standards** enforcement
- **Image vulnerability scanning** with Trivy
- **SAST** with CodeQL and Bandit
- **Dependency scanning** with Safety

### Access Control
- **IAM Roles for Service Accounts** (IRSA)
- **Basic auth** for monitoring endpoints
- **SSL certificates** via Let's Encrypt
- **API rate limiting** with nginx-ingress

## 🐛 Troubleshooting

### Common Issues

#### 1. Deployment Failures

```bash
# Check deployment status
kubectl get deployments -n rag-prod
kubectl describe deployment rag-backend -n rag-prod

# Check pod logs
kubectl logs -n rag-prod -l app=rag-backend --tail=100

# Check events
kubectl get events -n rag-prod --sort-by=.metadata.creationTimestamp
```

#### 2. Network Connectivity Issues

```bash
# Test internal connectivity
kubectl run debug --image=busybox -it --rm -- /bin/sh
nslookup rag-backend-service.rag-prod.svc.cluster.local

# Check ingress
kubectl get ingress -n rag-prod
kubectl describe ingress rag-production-ingress -n rag-prod
```

#### 3. Storage Issues

```bash
# Check PVCs
kubectl get pvc -n rag-prod
kubectl describe pvc rag-data-pvc -n rag-prod

# Check storage classes
kubectl get storageclass
```

#### 4. Scaling Issues

```bash
# Check HPA status
kubectl get hpa -n rag-prod
kubectl describe hpa rag-backend-hpa -n rag-prod

# Check metrics server
kubectl top nodes
kubectl top pods -n rag-prod
```

### Debugging Commands

```bash
# Complete cluster overview
kubectl cluster-info
kubectl get all -A

# Resource utilization
kubectl top nodes
kubectl get events --sort-by=.metadata.creationTimestamp -A

# Logs aggregation
kubectl logs -n rag-prod -l app=rag-backend -f --max-log-requests=10

# Network debugging
kubectl exec -it deployment/rag-backend -n rag-prod -- /bin/bash
```

### Recovery Procedures

#### 1. Application Recovery
```bash
# Restart deployment
kubectl rollout restart deployment/rag-backend -n rag-prod

# Rollback deployment
kubectl rollout undo deployment/rag-backend -n rag-prod

# Scale up/down
kubectl scale deployment/rag-backend --replicas=5 -n rag-prod
```

#### 2. Blue-Green Rollback
```bash
# Quick rollback to previous version
kubectl patch svc rag-app-active -n rag-prod -p '{"spec":{"selector":{"color":"blue"}}}'
```

#### 3. Database Recovery
```bash
# Restore from backup (example)
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier rag-prod-db-restored \
  --db-snapshot-identifier rag-prod-db-snapshot-2024-01-01
```

## 💰 Cost Optimization

### Development Environment
- **Spot instances**: 50-90% cost savings
- **Single NAT Gateway**: $32/month savings
- **Minimal resources**: t3.medium nodes
- **Short backup retention**: 7 days

### Production Cost Management
- **Reserved Instances**: Up to 75% savings for steady workloads
- **EBS GP3**: 20% cheaper than GP2 with better performance
- **S3 Intelligent Tiering**: Automatic cost optimization
- **CloudWatch optimization**: Custom metrics only when needed

### Monitoring Costs
```bash
# Check AWS costs by service
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Kubernetes resource requests vs usage
kubectl top nodes
kubectl top pods -A --sort-by=memory
```

### Cost Alerts
- **AWS Budgets** configured for each environment
- **Cost anomaly detection** enabled
- **Monthly cost reports** automated

## 📞 Support

### Getting Help
1. **Documentation**: Check this README and inline comments
2. **Monitoring**: Review Grafana dashboards and Prometheus alerts
3. **Logs**: Use Loki/Grafana or kubectl logs
4. **AWS Support**: For infrastructure issues
5. **GitHub Issues**: For CI/CD pipeline problems

### Emergency Contacts
- **Production Issues**: #production-alerts Slack channel
- **Infrastructure Team**: infrastructure@your-domain.com
- **On-call Rotation**: Check PagerDuty or similar

---

## 🎯 Next Steps

After successful deployment:

1. **Configure monitoring alerts** for your team
2. **Set up backup verification** procedures
3. **Plan capacity scaling** based on usage patterns
4. **Implement additional security measures** as needed
5. **Optimize costs** based on actual usage
6. **Set up disaster recovery** procedures
7. **Configure load testing** for capacity planning

For questions or issues, please refer to the troubleshooting section or contact the platform team.