pipeline {
    agent any

    stages {
        stage('Environment Setup') {
            steps {
                echo 'Setting up environment...'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow Pipeline...'
                sh 'python pipeline.py'
            }
        }
        
        stage('Verification') {
            steps {
                echo 'Verifying output...'
                sh 'ls -l pipeline.yaml'
            }
        }
    }
}