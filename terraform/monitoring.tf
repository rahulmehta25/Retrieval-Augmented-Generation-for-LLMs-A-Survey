# Monitoring Stack for RAG System

# Namespace for monitoring
resource "kubernetes_namespace" "monitoring" {
  count = var.enable_monitoring ? 1 : 0
  
  metadata {
    name = "monitoring"
    labels = {
      "app.kubernetes.io/name" = "monitoring"
    }
  }
}

# Prometheus Operator
resource "helm_release" "kube_prometheus_stack" {
  count = var.enable_monitoring ? 1 : 0
  
  name       = "kube-prometheus-stack"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "56.6.2"

  values = [
    yamlencode({
      # Prometheus configuration
      prometheus = {
        prometheusSpec = {
          retention = var.environment == "prod" ? "30d" : "7d"
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = var.environment == "prod" ? "50Gi" : "20Gi"
                  }
                }
                storageClassName = "gp2"
              }
            }
          }
          resources = {
            requests = {
              memory = var.environment == "prod" ? "2Gi" : "1Gi"
              cpu    = var.environment == "prod" ? "1000m" : "500m"
            }
            limits = {
              memory = var.environment == "prod" ? "4Gi" : "2Gi"
              cpu    = var.environment == "prod" ? "2000m" : "1000m"
            }
          }
          # Enable persistent storage
          persistentVolumeClaimRetentionPolicy = {
            whenDeleted = "Retain"
            whenScaled  = "Retain"
          }
          # Remote write configuration (optional for long-term storage)
          remoteWrite = var.environment == "prod" ? [
            {
              url = "https://your-prometheus-remote-write-endpoint"
              # Add authentication if needed
            }
          ] : []
        }
        service = {
          type = "LoadBalancer"
          annotations = {
            "service.beta.kubernetes.io/aws-load-balancer-type" = "nlb"
            "service.beta.kubernetes.io/aws-load-balancer-internal" = "true"
          }
        }
        ingress = {
          enabled = true
          ingressClassName = "nginx"
          annotations = {
            "cert-manager.io/cluster-issuer" = var.environment == "prod" ? "letsencrypt-prod" : "letsencrypt-staging"
            "nginx.ingress.kubernetes.io/auth-type" = "basic"
            "nginx.ingress.kubernetes.io/auth-secret" = "prometheus-basic-auth"
            "nginx.ingress.kubernetes.io/auth-realm" = "Authentication Required - Prometheus"
          }
          hosts = [
            {
              host = "prometheus${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"
              paths = [
                {
                  path = "/"
                  pathType = "Prefix"
                }
              ]
            }
          ]
          tls = [
            {
              secretName = "prometheus-tls"
              hosts = ["prometheus${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"]
            }
          ]
        }
      }
      
      # Grafana configuration
      grafana = {
        enabled = true
        adminPassword = random_password.grafana_admin[0].result
        persistence = {
          enabled = true
          size = "10Gi"
          storageClassName = "gp2"
        }
        resources = {
          requests = {
            memory = "256Mi"
            cpu = "100m"
          }
          limits = {
            memory = "512Mi"
            cpu = "500m"
          }
        }
        service = {
          type = "LoadBalancer"
          annotations = {
            "service.beta.kubernetes.io/aws-load-balancer-type" = "nlb"
            "service.beta.kubernetes.io/aws-load-balancer-internal" = "true"
          }
        }
        ingress = {
          enabled = true
          ingressClassName = "nginx"
          annotations = {
            "cert-manager.io/cluster-issuer" = var.environment == "prod" ? "letsencrypt-prod" : "letsencrypt-staging"
          }
          hosts = [
            "grafana${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"
          ]
          tls = [
            {
              secretName = "grafana-tls"
              hosts = ["grafana${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"]
            }
          ]
        }
        # Pre-configured dashboards
        dashboards = {
          default = {
            "kubernetes-cluster" = {
              gnetId = 7249
              revision = 1
              datasource = "Prometheus"
            }
            "kubernetes-pods" = {
              gnetId = 6417
              revision = 1
              datasource = "Prometheus"
            }
            "nginx-ingress" = {
              gnetId = 9614
              revision = 1
              datasource = "Prometheus"
            }
          }
        }
        # Data sources
        datasources = {
          "datasources.yaml" = {
            apiVersion = 1
            datasources = [
              {
                name = "Prometheus"
                type = "prometheus"
                url = "http://kube-prometheus-stack-prometheus:9090"
                access = "proxy"
                isDefault = true
              }
            ]
          }
        }
      }
      
      # Alertmanager configuration
      alertmanager = {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
                storageClassName = "gp2"
              }
            }
          }
          resources = {
            requests = {
              memory = "256Mi"
              cpu = "100m"
            }
            limits = {
              memory = "512Mi"
              cpu = "500m"
            }
          }
        }
        service = {
          type = "LoadBalancer"
          annotations = {
            "service.beta.kubernetes.io/aws-load-balancer-type" = "nlb"
            "service.beta.kubernetes.io/aws-load-balancer-internal" = "true"
          }
        }
        ingress = {
          enabled = true
          ingressClassName = "nginx"
          annotations = {
            "cert-manager.io/cluster-issuer" = var.environment == "prod" ? "letsencrypt-prod" : "letsencrypt-staging"
            "nginx.ingress.kubernetes.io/auth-type" = "basic"
            "nginx.ingress.kubernetes.io/auth-secret" = "alertmanager-basic-auth"
            "nginx.ingress.kubernetes.io/auth-realm" = "Authentication Required - Alertmanager"
          }
          hosts = [
            {
              host = "alertmanager${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"
              paths = [
                {
                  path = "/"
                  pathType = "Prefix"
                }
              ]
            }
          ]
          tls = [
            {
              secretName = "alertmanager-tls"
              hosts = ["alertmanager${var.environment != "prod" ? "-${var.environment}" : ""}.${var.domain_name}"]
            }
          ]
        }
        config = {
          global = {
            smtp_smarthost = "localhost:587"
            smtp_from = "alertmanager@${var.domain_name}"
          }
          route = {
            group_by = ["alertname"]
            group_wait = "10s"
            group_interval = "10s"
            repeat_interval = "1h"
            receiver = "web.hook"
          }
          receivers = [
            {
              name = "web.hook"
              slack_configs = var.slack_webhook_url != "" ? [
                {
                  api_url = var.slack_webhook_url
                  channel = "#alerts"
                  title = "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                  text = "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                }
              ] : []
              email_configs = [
                {
                  to = "alerts@${var.domain_name}"
                  subject = "{{ .GroupLabels.alertname }}"
                  body = "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                }
              ]
            }
          ]
        }
      }
      
      # Node Exporter
      nodeExporter = {
        enabled = true
      }
      
      # Kube State Metrics
      kubeStateMetrics = {
        enabled = true
      }
    })
  ]

  depends_on = [kubernetes_namespace.monitoring]
}

# Generate random password for Grafana admin
resource "random_password" "grafana_admin" {
  count = var.enable_monitoring ? 1 : 0
  
  length  = 16
  special = true
}

# Store Grafana admin password in AWS Secrets Manager
resource "aws_secretsmanager_secret" "grafana_admin" {
  count = var.enable_monitoring ? 1 : 0
  
  name = "${local.cluster_name}-grafana-admin-password"
  description = "Grafana admin password for ${local.cluster_name}"

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "grafana_admin" {
  count = var.enable_monitoring ? 1 : 0
  
  secret_id = aws_secretsmanager_secret.grafana_admin[0].id
  secret_string = jsonencode({
    username = "admin"
    password = random_password.grafana_admin[0].result
  })
}

# Basic auth secrets for Prometheus and Alertmanager
resource "kubernetes_secret" "prometheus_basic_auth" {
  count = var.enable_monitoring ? 1 : 0
  
  metadata {
    name      = "prometheus-basic-auth"
    namespace = "monitoring"
  }

  type = "Opaque"

  data = {
    auth = base64encode("admin:${bcrypt(random_password.prometheus_auth[0].result)}")
  }
}

resource "kubernetes_secret" "alertmanager_basic_auth" {
  count = var.enable_monitoring ? 1 : 0
  
  metadata {
    name      = "alertmanager-basic-auth"
    namespace = "monitoring"
  }

  type = "Opaque"

  data = {
    auth = base64encode("admin:${bcrypt(random_password.alertmanager_auth[0].result)}")
  }
}

resource "random_password" "prometheus_auth" {
  count = var.enable_monitoring ? 1 : 0
  
  length  = 16
  special = false
}

resource "random_password" "alertmanager_auth" {
  count = var.enable_monitoring ? 1 : 0
  
  length  = 16
  special = false
}

# Loki for log aggregation
resource "helm_release" "loki" {
  count = var.enable_logging ? 1 : 0
  
  name       = "loki"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "loki"
  namespace  = "monitoring"
  version    = "5.41.4"

  values = [
    yamlencode({
      loki = {
        auth_enabled = false
        commonConfig = {
          replication_factor = var.environment == "prod" ? 3 : 1
        }
        storage = {
          type = "s3"
          bucketNames = {
            chunks = aws_s3_bucket.loki_chunks[0].bucket
            ruler = aws_s3_bucket.loki_chunks[0].bucket
            admin = aws_s3_bucket.loki_chunks[0].bucket
          }
          s3 = {
            region = var.aws_region
            endpoint = "s3.${var.aws_region}.amazonaws.com"
            s3ForcePathStyle = false
          }
        }
      }
      write = {
        replicas = var.environment == "prod" ? 3 : 1
        persistence = {
          size = "10Gi"
          storageClass = "gp2"
        }
        resources = {
          requests = {
            memory = "512Mi"
            cpu = "200m"
          }
          limits = {
            memory = "1Gi"
            cpu = "500m"
          }
        }
      }
      read = {
        replicas = var.environment == "prod" ? 3 : 1
        persistence = {
          size = "10Gi"
          storageClass = "gp2"
        }
        resources = {
          requests = {
            memory = "512Mi"
            cpu = "200m"
          }
          limits = {
            memory = "1Gi"
            cpu = "500m"
          }
        }
      }
      backend = {
        replicas = var.environment == "prod" ? 3 : 1
        persistence = {
          size = "10Gi"
          storageClass = "gp2"
        }
        resources = {
          requests = {
            memory = "512Mi"
            cpu = "200m"
          }
          limits = {
            memory = "1Gi"
            cpu = "500m"
          }
        }
      }
      serviceAccount = {
        annotations = {
          "eks.amazonaws.com/role-arn" = aws_iam_role.loki[0].arn
        }
      }
    })
  ]

  depends_on = [
    kubernetes_namespace.monitoring,
    aws_s3_bucket.loki_chunks
  ]
}

# S3 bucket for Loki chunks
resource "aws_s3_bucket" "loki_chunks" {
  count = var.enable_logging ? 1 : 0
  
  bucket = "${local.cluster_name}-loki-chunks-${random_id.loki_bucket_suffix[0].hex}"

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-loki-chunks"
  })
}

resource "random_id" "loki_bucket_suffix" {
  count = var.enable_logging ? 1 : 0
  
  byte_length = 8
}

resource "aws_s3_bucket_versioning" "loki_chunks" {
  count = var.enable_logging ? 1 : 0
  
  bucket = aws_s3_bucket.loki_chunks[0].id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "loki_chunks" {
  count = var.enable_logging ? 1 : 0
  
  bucket = aws_s3_bucket.loki_chunks[0].id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.eks.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "loki_chunks" {
  count = var.enable_logging ? 1 : 0
  
  bucket = aws_s3_bucket.loki_chunks[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM role for Loki
resource "aws_iam_role" "loki" {
  count = var.enable_logging ? 1 : 0
  
  name = "${local.cluster_name}-loki-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:monitoring:loki"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "loki" {
  count = var.enable_logging ? 1 : 0
  
  name = "${local.cluster_name}-loki-policy"
  role = aws_iam_role.loki[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.loki_chunks[0].arn,
          "${aws_s3_bucket.loki_chunks[0].arn}/*"
        ]
      }
    ]
  })
}

# Promtail for log collection
resource "helm_release" "promtail" {
  count = var.enable_logging ? 1 : 0
  
  name       = "promtail"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "promtail"
  namespace  = "monitoring"
  version    = "6.15.3"

  values = [
    yamlencode({
      config = {
        clients = [
          {
            url = "http://loki-gateway/loki/api/v1/push"
          }
        ]
        snippets = {
          scrapeConfigs = |
            # Pod logs
            - job_name: kubernetes-pods
              kubernetes_sd_configs:
                - role: pod
              pipeline_stages:
                - cri: {}
              relabel_configs:
                - source_labels:
                    - __meta_kubernetes_pod_controller_name
                  regex: ([0-9a-z-.]+?)(-[0-9a-f]{8,10})?
                  action: replace
                  target_label: __tmp_controller_name
                - source_labels:
                    - __meta_kubernetes_pod_label_app_kubernetes_io_name
                    - __meta_kubernetes_pod_label_app
                    - __tmp_controller_name
                    - __meta_kubernetes_pod_name
                  regex: ^;*([^;]+)(;.*)?$
                  action: replace
                  target_label: app
                - source_labels:
                    - __meta_kubernetes_pod_label_app_kubernetes_io_instance
                    - __meta_kubernetes_pod_label_release
                  regex: ^;*([^;]+)(;.*)?$
                  action: replace
                  target_label: instance
                - source_labels:
                    - __meta_kubernetes_pod_label_app_kubernetes_io_component
                    - __meta_kubernetes_pod_label_component
                  regex: ^;*([^;]+)(;.*)?$
                  action: replace
                  target_label: component
      })
      resources = {
        limits = {
          cpu = "200m"
          memory = "128Mi"
        }
        requests = {
          cpu = "100m"
          memory = "128Mi"
        }
      }
    })
  ]

  depends_on = [
    helm_release.loki,
    kubernetes_namespace.monitoring
  ]
}

# CloudWatch integration (optional)
resource "helm_release" "cloudwatch_metrics" {
  count = var.enable_monitoring && var.environment == "prod" ? 1 : 0
  
  name       = "cloudwatch-metrics"
  repository = "https://kubernetes-sigs.github.io/aws-cloudwatch-metrics"
  chart      = "aws-cloudwatch-metrics"
  namespace  = "amazon-cloudwatch"
  version    = "0.0.9"

  create_namespace = true

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "awsRegion"
    value = var.aws_region
  }
}