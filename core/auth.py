import json
import os
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt

# Local-only security settings
DB_PATH = "db/users.json"
SECRET_KEY = "local-secret-for-lexai-security" # Only for local JWT signature
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    def __init__(self):
        self._ensure_db()
        self.users = self._load_users()

    def _ensure_db(self):
        os.makedirs("db", exist_ok=True)
        if not os.path.exists(DB_PATH):
            with open(DB_PATH, 'w') as f:
                json.dump({}, f)

    def _load_users(self):
        with open(DB_PATH, 'r') as f:
            return json.load(f)

    def _save_users(self):
        with open(DB_PATH, 'w') as f:
            json.dump(self.users, f, indent=4)

    def hash_password(self, password):
        return pwd_context.hash(password)

    def verify_password(self, password, hashed):
        return pwd_context.verify(password, hashed)

    def signup(self, username, password, role="employee"):
        if username in self.users:
            return False, "Username already exists"
        
        # Valid roles
        valid_roles = ["owner", "accountant", "sales_staff", "employee"]
        if role not in valid_roles:
            role = "employee"

        self.users[username] = {
            "password": self.hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat()
        }
        self._save_users()
        return True, "Account created successfully"

    def login(self, username, password):
        user = self.users.get(username)
        if not user or not self.verify_password(password, user["password"]):
            return None, "Invalid username or password"
        
        # Create a local session token with role
        expiration = datetime.now() + timedelta(days=7)
        token_data = {
            "sub": username, 
            "role": user.get("role", "employee"),
            "exp": expiration
        }
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        return token, "Logged in successfully"

_auth_instance = None

def get_auth_manager():
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = AuthManager()
    return _auth_instance
