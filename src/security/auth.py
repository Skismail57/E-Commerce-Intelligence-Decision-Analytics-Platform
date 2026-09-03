"""
Security and Governance Module
Implements JWT authentication and RBAC for security and governance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import secrets
from config.logging_config import get_logger

logger = get_logger(__name__)


class AuthManager:
    """
    Authentication and authorization manager.
    
    Features:
    - JWT token generation and validation
    - Role-Based Access Control (RBAC)
    - User management
    - Permission checking
    - Session management
    """
    
    def __init__(self, secret_key: str = None):
        """
        Initialize auth manager.
        
        Args:
            secret_key: Secret key for JWT signing
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users = {}
        self.roles = {}
        self.permissions = {}
        self.sessions = {}
        
        # Initialize default roles and permissions
        self._initialize_default_roles()
        
        logger.info("Auth manager initialized")
    
    def _initialize_default_roles(self):
        """Initialize default roles and permissions."""
        self.roles = {
            'admin': {
                'name': 'Administrator',
                'permissions': ['read', 'write', 'delete', 'admin', 'deploy', 'train']
            },
            'analyst': {
                'name': 'Data Analyst',
                'permissions': ['read', 'write']
            },
            'viewer': {
                'name': 'Viewer',
                'permissions': ['read']
            }
        }
        
        self.permissions = {
            'read': 'Read data and models',
            'write': 'Write data and models',
            'delete': 'Delete data and models',
            'admin': 'Administrative functions',
            'deploy': 'Deploy models to production',
            'train': 'Train new models'
        }
    
    def create_user(
        self,
        username: str,
        password: str,
        role: str = 'viewer',
        email: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Create a new user.
        
        Args:
            username: Username
            password: Password (will be hashed)
            role: User role
            email: User email
            metadata: Additional user metadata
        
        Returns:
            Dictionary with user information
        """
        logger.info(f"Creating user: {username}")
        
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        if role not in self.roles:
            raise ValueError(f"Role {role} does not exist")
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Create user
        user = {
            'username': username,
            'password_hash': password_hash,
            'role': role,
            'email': email,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'is_active': True
        }
        
        self.users[username] = user
        
        logger.info(f"User {username} created with role {role}")
        
        return user
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, username: str, password: str) -> bool:
        """
        Verify user password.
        
        Args:
            username: Username
            password: Password to verify
        
        Returns:
            True if password is correct
        """
        if username not in self.users:
            return False
        
        password_hash = self._hash_password(password)
        return self.users[username]['password_hash'] == password_hash
    
    def generate_token(
        self,
        username: str,
        expires_in_hours: int = 24
    ) -> str:
        """
        Generate JWT token for user.
        
        Args:
            username: Username
            expires_in_hours: Token expiration time in hours
        
        Returns:
            JWT token string
        """
        logger.info(f"Generating token for user: {username}")
        
        if username not in self.users:
            raise ValueError(f"User {username} not found")
        
        user = self.users[username]
        
        # Create token payload
        payload = {
            'username': username,
            'role': user['role'],
            'permissions': self.roles[user['role']]['permissions'],
            'iat': datetime.now().timestamp(),
            'exp': (datetime.now() + timedelta(hours=expires_in_hours)).timestamp()
        }
        
        # Sign token (simplified JWT implementation)
        token = self._sign_token(payload)
        
        # Store session
        self.sessions[token] = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=expires_in_hours)).isoformat()
        }
        
        logger.info(f"Token generated for user {username}")
        
        return token
    
    def _sign_token(self, payload: Dict) -> str:
        """Sign token payload."""
        import json
        import base64
        
        # Encode payload
        payload_str = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_str.encode()).decode()
        
        # Create signature
        signature_data = f"{payload_b64}.{self.secret_key}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        # Return token
        token = f"{payload_b64}.{signature}"
        
        return token
    
    def validate_token(self, token: str) -> Dict:
        """
        Validate JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            Dictionary with token payload if valid
        """
        import json
        import base64
        
        # Check if token exists in sessions
        if token not in self.sessions:
            return {'valid': False, 'reason': 'Token not found'}
        
        session = self.sessions[token]
        
        # Check expiration
        expires_at = datetime.fromisoformat(session['expires_at'])
        if datetime.now() > expires_at:
            return {'valid': False, 'reason': 'Token expired'}
        
        # Decode token
        try:
            payload_b64, signature = token.split('.')
            payload_str = base64.urlsafe_b64decode(payload_b64).decode()
            payload = json.loads(payload_str)
            
            # Verify signature
            signature_data = f"{payload_b64}.{self.secret_key}"
            expected_signature = hashlib.sha256(signature_data.encode()).hexdigest()
            
            if signature != expected_signature:
                return {'valid': False, 'reason': 'Invalid signature'}
            
            # Check expiration
            if datetime.now().timestamp() > payload['exp']:
                return {'valid': False, 'reason': 'Token expired'}
            
            return {'valid': True, 'payload': payload}
        
        except Exception as e:
            return {'valid': False, 'reason': str(e)}
    
    def check_permission(
        self,
        username: str,
        permission: str
    ) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            username: Username
            permission: Permission to check
        
        Returns:
            True if user has permission
        """
        if username not in self.users:
            return False
        
        user = self.users[username]
        role_permissions = self.roles[user['role']]['permissions']
        
        return permission in role_permissions
    
    def check_permissions(
        self,
        username: str,
        permissions: List[str]
    ) -> Dict:
        """
        Check if user has multiple permissions.
        
        Args:
            username: Username
            permissions: List of permissions to check
        
        Returns:
            Dictionary with permission check results
        """
        results = {}
        
        for permission in permissions:
            results[permission] = self.check_permission(username, permission)
        
        return results
    
    def assign_role(
        self,
        username: str,
        new_role: str
    ) -> Dict:
        """
        Assign new role to user.
        
        Args:
            username: Username
            new_role: New role to assign
        
        Returns:
            Dictionary with assignment result
        """
        logger.info(f"Assigning role {new_role} to user {username}")
        
        if username not in self.users:
            raise ValueError(f"User {username} not found")
        
        if new_role not in self.roles:
            raise ValueError(f"Role {new_role} does not exist")
        
        self.users[username]['role'] = new_role
        self.users[username]['updated_at'] = datetime.now().isoformat()
        
        logger.info(f"Role {new_role} assigned to user {username}")
        
        return self.users[username]
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token.
        
        Args:
            token: Token to revoke
        
        Returns:
            True if token was revoked
        """
        if token in self.sessions:
            del self.sessions[token]
            logger.info("Token revoked")
            return True
        return False
    
    def list_users(self) -> List[Dict]:
        """
        List all users.
        
        Returns:
            List of user information
        """
        users = []
        
        for username, user_info in self.users.items():
            users.append({
                'username': username,
                'role': user_info['role'],
                'email': user_info.get('email'),
                'is_active': user_info['is_active'],
                'created_at': user_info['created_at']
            })
        
        return users
    
    def list_roles(self) -> Dict:
        """
        List all roles and their permissions.
        
        Returns:
            Dictionary with role information
        """
        return self.roles
    
    def get_user_permissions(self, username: str) -> List[str]:
        """
        Get permissions for a user.
        
        Args:
            username: Username
        
        Returns:
            List of permissions
        """
        if username not in self.users:
            return []
        
        role = self.users[username]['role']
        return self.roles[role]['permissions']


def run_auth_pipeline(
    username: str,
    password: str,
    role: str = 'analyst'
) -> Tuple[AuthManager, Dict]:
    """
    Convenience function to run auth pipeline.
    
    Args:
        username: Username
        password: Password
        role: User role
    
    Returns:
        Tuple of (auth_manager, results)
    """
    auth = AuthManager()
    
    # Create user
    user = auth.create_user(username, password, role)
    
    # Generate token
    token = auth.generate_token(username)
    
    # Validate token
    validation = auth.validate_token(token)
    
    # Check permissions
    permissions = auth.get_user_permissions(username)
    
    results = {
        'user': user,
        'token': token,
        'validation': validation,
        'permissions': permissions
    }
    
    return auth, results
