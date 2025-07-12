#!/bin/bash

# GPU Node Setup Automation Script
# This script sets up a new GPU node server for model training

set -e  # Exit on any error

echo "🚀 Starting GPU Node Setup..."
echo "=================================="

# Configuration variables - Edit these as needed
STORAGE_BASE_URL="/home/ubuntu/porialab-us-south-2"
GIT_USERNAME="ob1-brandon"
GIT_EMAIL="ob1-brandon@aisg.org"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# # Install essential packages
# print_status "Installing essential packages..."

# # 1. Install AWS CLI (latest version)
# print_status "Installing AWS CLI..."
# if command_exists aws; then
#     print_warning "AWS CLI already installed, skipping..."
# else
#     curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#     unzip awscliv2.zip
#     sudo ./aws/install
#     rm -rf awscliv2.zip aws/
# fi

# # Verify AWS CLI installation
# if command_exists aws; then
#     print_status "AWS CLI installed successfully: $(aws --version)"
# else
#     print_error "AWS CLI installation failed"
#     exit 1
# fi

# # 4. Install uv (fast Python package manager)
# print_status "Installing uv..."
# if command_exists uv; then
#     print_warning "uv already installed, skipping..."
# else
#     curl -LsSf https://astral.sh/uv/install.sh | sh
#     source ~/.bashrc
# fi

# # Activate virtual environment
# uv venv
# source .venv/bin/activate

# # 2. Install Wandb CLI
# print_status "Installing Wandb CLI..."
# if command_exists wandb; then
#     print_warning "Wandb CLI already installed, skipping..."
# else
#     uv pip install wandb
# fi

# # 3. Install Hugging Face CLI
# print_status "Installing Hugging Face CLI..."
# if command_exists huggingface-cli; then
#     print_warning "Hugging Face CLI already installed, skipping..."
# else
#     uv pip install huggingface_hub
# fi

# # Install requirements if requirements.txt exists
# if [ -f "requirements.txt" ]; then
#     print_status "Installing requirements from requirements.txt..."
#     if command_exists uv; then
#         uv pip install -r requirements.txt
#     else
#         pip install -r requirements.txt
#     fi
# else
#     print_warning "requirements.txt not found, skipping package installation"
# fi

# 6. Create .bashrc shortcuts
print_status "Setting up .bashrc shortcuts..."

# Validate configuration variables
if [ -z "$STORAGE_BASE_URL" ]; then
    print_error "STORAGE_BASE_URL is not set. Please edit the script and set it at the top."
    exit 1
fi

# Create backup of existing .bashrc
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    print_status "Backup of existing .bashrc created"
fi

# Add custom shortcuts to .bashrc
cat >> ~/.bashrc << 'EOL'

# Custom shortcuts for GPU node setup
alias sov='source .venv/bin/activate'
alias gfp='git add . && git commit -m "gfp" && git push'

EOL

# Add the cob1 alias with the user-provided path
echo "alias cob1='cd ${STORAGE_BASE_URL}/ntu/brandon'" >> ~/.bashrc

print_status "Custom shortcuts added to .bashrc"

# 7. Set up Git configuration
print_status "Setting up Git configuration..."

# Validate Git configuration variables
if [ -z "$GIT_USERNAME" ] || [ -z "$GIT_EMAIL" ]; then
    print_error "GIT_USERNAME or GIT_EMAIL is not set. Please edit the script and set them at the top."
    exit 1
fi

# Configure Git
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_EMAIL"

print_status "Git configuration set:"
print_status "  Username: $(git config --global user.name)"
print_status "  Email: $(git config --global user.email)"

# Test AWS CLI authentication
print_status "Testing AWS CLI authentication..."
print_status "Please run 'aws configure' to set up your credentials after this script completes"

# Final steps
print_status "=================================="
print_status "🎉 GPU Node Setup Complete!"
print_status "=================================="

echo ""
print_status "Next steps:"
echo "1. Configure AWS credentials by running: aws configure"
echo "2. Login to Wandb by running: wandb login"
echo "3. Login to Hugging Face by running: huggingface-cli login"
echo "4. Reload your shell: source ~/.bashrc"
echo "5. Test your shortcuts:"
echo "   - sov (activate virtual environment)"
echo "   - cob1 (navigate to your project directory)"
echo "   - gfp (git add, commit, push)"

print_status "Your original .bashrc has been backed up"
print_status "Setup completed successfully! 🚀"