# Using EC2 instances

## Launching an instance

Navigate to the EC2 instances page on AWS and select launch instance. Choose platform Deep Learning Ubuntu 18.04. Choose instance type p2.xlarge. For tags, add project_group=Lyft 3d, student_name={your name}, and Name={readable name of EC2 instance, e.g. lyft3d-ec2}. Also add a key pair (create if necessary) so that you can SSH into the instance. Launch.

## SSH an instance

When the instance has completed setup and is in the running state, SSH into the instance with the command below. The public IPv4 DNS is available on the EC2 instances page.

```
ssh -i {key.pem file you used in setup} ec2-user@{EC2 instance public IPv4 DNS}
```
