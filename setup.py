import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🤖 Askademia Setup")
    print("made by Tuhin ❤️")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    print("\n🏗️ Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_pip_command():
    """Get the appropriate pip command for the platform"""
    if platform.system() == "Windows":
        return ["venv\\Scripts\\pip"]
    else:
        return ["venv/bin/pip"]

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        print("Creating basic requirements.txt...")
        
        requirements = """# Core Dependencies
streamlit>=1.28.0
openai>=1.3.0
tiktoken>=0.5.0

# PDF Processing
PyPDF2>=3.0.1
PyMuPDF>=1.23.0
pdfplumber>=0.9.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# ArXiv Integration
arxiv>=1.4.0
requests>=2.31.0

# Development Dependencies (optional)
reportlab>=3.6.0
"""
        
        with open("requirements.txt", "w") as f:
            f.write(requirements)
        
        print("✅ requirements.txt created")
    
    try:
        pip_cmd = get_pip_command()
        python_cmd = ["venv\\Scripts\\python"] if platform.system() == "Windows" else ["venv/bin/python"]

        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run(python_cmd + ["-m", "pip", "install", "--upgrade", "pip"], check=True)

        # Install requirements
        print("Installing packages...")
        subprocess.run(python_cmd + ["-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_project_structure():
    """Create necessary project directories"""
    print("\n📁 Creating project structure...")
    
    directories = [
        "documents",
        "downloads",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create environment file template"""
    print("\n🔧 Creating environment configuration...")
    
    env_content = """# OpenAI API Configuration
# OPENAI_API_KEY=your_api_key_here
# OPENAI_MODEL=gpt-3.5-turbo

# Application Configuration
# MAX_FILE_SIZE=50
# LOG_LEVEL=INFO

# Security Configuration
# RATE_LIMIT_CALLS=60
# RATE_LIMIT_PERIOD=60
"""
    
    env_file = Path(".env.example")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env.example file")
        print("   Copy to .env and add your OpenAI API key")
    else:
        print("✅ .env.example already exists")

def validate_installation():
    """Validate the installation"""
    print("\n🧪 Validating installation...")
    
    try:
        # Try to import key modules
        pip_cmd = get_pip_command()
        
        # Get installed packages
        result = subprocess.run(pip_cmd + ["list"], 
                              capture_output=True, text=True, check=True)
        
        required_packages = [
            'streamlit', 'openai', 'tiktoken', 'PyPDF2', 
            'PyMuPDF', 'pdfplumber', 'pandas', 'arxiv', 'requests'
        ]
        
        installed = result.stdout.lower()
        
        for package in required_packages:
            if package.lower() in installed:
                print(f"✅ {package}")
            else:
                print(f"❌ {package} - Not found")
                return False
        
        print("✅ All packages validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def create_launch_scripts():
    """Create convenient launch scripts"""
    print("\n🚀 Creating launch scripts...")
    
    # Windows batch file
    windows_script = """@echo off
echo Starting Document Q&A AI Agent...
call venv\\Scripts\\activate
streamlit run app.py
pause
"""
    
    with open("start_agent.bat", "w") as f:
        f.write(windows_script)
    print("✅ Created start_agent.bat (Windows)")
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting Document Q&A AI Agent..."
source venv/bin/activate
streamlit run app.py
"""
    
    with open("start_agent.sh", "w") as f:
        f.write(unix_script)
    
    # Make shell script executable
    if platform.system() != "Windows":
        os.chmod("start_agent.sh", 0o755)
    
    print("✅ Created start_agent.sh (Unix/Linux/macOS)")

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running test suite...")
    
    if not Path("test_agent.py").exists():
        print("⚠️ test_agent.py not found - skipping tests")
        return True
    
    try:
        if platform.system() == "Windows":
            python_cmd = "venv\\Scripts\\python"
        else:
            python_cmd = "venv/bin/python"
        
        subprocess.run([python_cmd, "test_agent.py"], check=True)
        print("✅ Test suite passed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Some tests failed: {e}")
        print("This is normal if OpenAI API key is not configured")
        return True

def print_completion_message():
    """Print completion message with next steps"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. 🔑 Get an OpenAI API key from https://platform.openai.com/")
    print("2. 📝 Create .env file: cp .env.example .env")
    print("3. ✏️ Add your API key to .env file")
    print("4. 🚀 Start the application:")
    
    if platform.system() == "Windows":
        print("   • Double-click start_agent.bat")
        print("   • Or run: venv\\Scripts\\activate && streamlit run app.py")
    else:
        print("   • Run: ./start_agent.sh")
        print("   • Or run: source venv/bin/activate && streamlit run app.py")
    
    print("\n🌐 The application will open in your browser at http://localhost:8501")
    
    print("\n🔧 Configuration:")
    print("   • Upload PDF documents using the web interface")
    print("   • Enter your OpenAI API key in the sidebar")
    print("   • Start asking questions about your documents!")
    
    print("\n📚 Documentation:")
    print("   • README.md - Complete usage guide")
    print("   • test_agent.py - Test suite and examples")
    print("   • main.py - Core application code")

def main():
    """Main setup function"""
    print_header()
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Setup failed - could not create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed - could not install dependencies")
        sys.exit(1)
    
    # Create project structure
    create_project_structure()
    
    # Create configuration files
    create_env_file()
    
    # Validate installation
    if not validate_installation():
        print("❌ Setup completed with warnings - some packages may be missing")
    
    # Create launch scripts
    create_launch_scripts()
    
    # Run tests (optional)
    run_tests()
    
    # Show completion message
    print_completion_message()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)