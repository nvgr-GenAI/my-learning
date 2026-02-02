# Infrastructure as Code

**Manage infrastructure through code** | 🏗️ Terraform | ☁️ CloudFormation | 🔧 Ansible

---

## Overview

Infrastructure as Code (IaC) treats infrastructure configuration as software code, enabling version control, automated provisioning, and consistent environments.

**Benefits:**
- Repeatable deployments
- Version controlled infrastructure
- Reduced human error
- Faster provisioning

---

## Terraform

=== "Basics"
    ```hcl
    # provider.tf
    terraform {
      required_version = ">= 1.0"
      required_providers {
        aws = {
          source  = "hashicorp/aws"
          version = "~> 4.0"
        }
      }
      
      backend "s3" {
        bucket = "myapp-terraform-state"
        key    = "production/terraform.tfstate"
        region = "us-east-1"
        dynamodb_table = "terraform-locks"
        encrypt = true
      }
    }

    provider "aws" {
      region = var.aws_region
      
      default_tags {
        tags = {
          Environment = var.environment
          Project     = "myapp"
          ManagedBy   = "Terraform"
        }
      }
    }
    ```

=== "Complete Example"
    ```hcl
    # variables.tf
    variable "aws_region" {
      description = "AWS region"
      type        = string
      default     = "us-east-1"
    }

    variable "environment" {
      description = "Environment name"
      type        = string
      validation {
        condition     = contains(["dev", "staging", "production"], var.environment)
        error_message = "Environment must be dev, staging, or production."
      }
    }

    variable "instance_count" {
      description = "Number of EC2 instances"
      type        = number
      default     = 2
    }

    # main.tf
    # VPC
    resource "aws_vpc" "main" {
      cidr_block           = "10.0.0.0/16"
      enable_dns_hostnames = true
      enable_dns_support   = true
    }

    # Subnets
    resource "aws_subnet" "public" {
      count                   = 2
      vpc_id                  = aws_vpc.main.id
      cidr_block              = "10.0.${count.index}.0/24"
      availability_zone       = data.aws_availability_zones.available.names[count.index]
      map_public_ip_on_launch = true
    }

    # Internet Gateway
    resource "aws_internet_gateway" "main" {
      vpc_id = aws_vpc.main.id
    }

    # Security Group
    resource "aws_security_group" "web" {
      name        = "${var.environment}-web-sg"
      description = "Allow HTTP/HTTPS inbound traffic"
      vpc_id      = aws_vpc.main.id

      ingress {
        description = "HTTPS from anywhere"
        from_port   = 443
        to_port     = 443
        protocol    = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
      }

      ingress {
        description = "HTTP from anywhere"
        from_port   = 80
        to_port     = 80
        protocol    = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
      }

      egress {
        from_port   = 0
        to_port     = 0
        protocol    = "-1"
        cidr_blocks = ["0.0.0.0/0"]
      }
    }

    # EC2 Instances
    resource "aws_instance" "web" {
      count         = var.instance_count
      ami           = data.aws_ami.ubuntu.id
      instance_type = "t3.micro"
      subnet_id     = aws_subnet.public[count.index % 2].id
      
      vpc_security_group_ids = [aws_security_group.web.id]

      user_data = <<-EOF
                  #!/bin/bash
                  apt-get update
                  apt-get install -y nginx
                  systemctl start nginx
                  systemctl enable nginx
                  EOF

      tags = {
        Name = "${var.environment}-web-${count.index}"
      }
    }

    # Load Balancer
    resource "aws_lb" "main" {
      name               = "${var.environment}-lb"
      internal           = false
      load_balancer_type = "application"
      security_groups    = [aws_security_group.web.id]
      subnets            = aws_subnet.public[*].id
    }

    # Target Group
    resource "aws_lb_target_group" "web" {
      name     = "${var.environment}-web-tg"
      port     = 80
      protocol = "HTTP"
      vpc_id   = aws_vpc.main.id

      health_check {
        path                = "/"
        healthy_threshold   = 2
        unhealthy_threshold = 10
        timeout             = 5
        interval            = 30
      }
    }

    # Attach instances to target group
    resource "aws_lb_target_group_attachment" "web" {
      count            = var.instance_count
      target_group_arn = aws_lb_target_group.web.arn
      target_id        = aws_instance.web[count.index].id
      port             = 80
    }

    # outputs.tf
    output "load_balancer_dns" {
      description = "DNS name of load balancer"
      value       = aws_lb.main.dns_name
    }

    output "instance_ips" {
      description = "Public IPs of instances"
      value       = aws_instance.web[*].public_ip
    }
    ```

=== "Commands"
    ```bash
    # Initialize
    terraform init

    # Plan changes
    terraform plan

    # Apply changes
    terraform apply

    # Apply with variables
    terraform apply -var="environment=production"

    # Destroy infrastructure
    terraform destroy

    # Format code
    terraform fmt

    # Validate configuration
    terraform validate

    # Show current state
    terraform show

    # List resources
    terraform state list

    # Import existing resource
    terraform import aws_instance.web i-1234567890abcdef0
    ```

---

## AWS CloudFormation

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Web application infrastructure'

Parameters:
  EnvironmentName:
    Description: Environment name
    Type: String
    AllowedValues:
      - dev
      - staging
      - production
    Default: dev

  InstanceType:
    Description: EC2 instance type
    Type: String
    Default: t3.micro
    AllowedValues:
      - t3.micro
      - t3.small
      - t3.medium

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-vpc

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-igw

  AttachGateway:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      VpcId: !Ref VPC
      InternetGatewayId: !Ref InternetGateway

  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true

  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Enable HTTP/HTTPS access
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

  WebServerInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: !FindInMap [RegionMap, !Ref 'AWS::Region', AMI]
      SecurityGroupIds:
        - !Ref WebServerSecurityGroup
      SubnetId: !Ref PublicSubnet1
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          yum install -y httpd
          systemctl start httpd
          systemctl enable httpd

Outputs:
  WebServerPublicIP:
    Description: Public IP of web server
    Value: !GetAtt WebServerInstance.PublicIp
    Export:
      Name: !Sub ${EnvironmentName}-WebServerIP
```

---

## Interview Talking Points

**Q: Terraform vs CloudFormation - when to use each?**

✅ **Strong Answer:**
> "I'd use Terraform for multi-cloud deployments or when we need a unified tool across AWS, GCP, and Azure. Terraform's declarative syntax and large provider ecosystem make it versatile. However, I'd use CloudFormation for AWS-only infrastructure since it's native to AWS, has deeper integration with AWS services, and doesn't require managing Terraform state files. CloudFormation also supports drift detection and has built-in rollback capabilities. The choice depends on whether we're locked into AWS or need multi-cloud flexibility."

---

## Related Topics

- [Containers](containers.md) - Provision container infrastructure
- [CI/CD](ci-cd.md) - Automate infrastructure deployment
- [Deployment Strategies](strategies.md) - Deploy with IaC

---

**Infrastructure as code, everything versioned! 🏗️**
