# reference: https://fastapi.tiangolo.com/tutorial/security/

import logging
import os

import uvicorn
import yaml

# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from pydantic import BaseModel
from src.inference.inference_class import Inference

from fastapi import FastAPI, HTTPException  # Depends,; status

# from datetime import datetime, timedelta, timezone
# from typing import Annotated

CONFIG_PATH = './inference_config.yml'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


# # FastAPI authentication with OAuth2
# # to get a string like this run:
# # openssl rand -hex 32
# # TODO: replace with SECRET_KEY from k8s secret and load as env var
# # used for encoding the access token for
# # the user chat client (used in HTTP request header)
# SECRET_KEY = "<secret key>"
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 30


# # TODO: replace db for OAuth2 authentication with a real db and
# # only load db for user_id_ (read from env var above)
# fake_users_db = {
#     "johndoe": {
#         "username": "johndoe",
#         "full_name": "John Doe",
#         "email": "johndoe@example.com",
#         "hashed_password":
#         "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
#         "disabled": False,
#     }
# }


# class Token(BaseModel):
#     access_token: str
#     token_type: str


# class TokenData(BaseModel):
#     username: str | None = None


# class User(BaseModel):
#     username: str
#     email: str | None = None
#     full_name: str | None = None
#     disabled: bool | None = None


# class UserInDB(User):
#     hashed_password: str


# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)


# def get_password_hash(password):
#     return pwd_context.hash(password)


# def get_user(db, username: str):
#     if username in db:
#         user_dict = db[username]
#         return UserInDB(**user_dict)


# def authenticate_user(fake_db, username: str, password: str):
#     user = get_user(fake_db, username)
#     if not user:
#         return False
#     if not verify_password(password, user.hashed_password):
#         return False
#     return user


# def create_access_token(data: dict, expires_delta: timedelta | None = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.now(timezone.utc) + expires_delta
#     else:
#         expire = datetime.now(timezone.utc) + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt


# async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#         token_data = TokenData(username=username)
#     except JWTError:
#         raise credentials_exception
#     user = get_user(fake_users_db, username=token_data.username)
#     if user is None:
#         raise credentials_exception
#     return user


# async def get_current_active_user(
#     current_user: Annotated[User, Depends(get_current_user)]
# ):
#     if current_user.disabled:
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user


global config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, yaml.SafeLoader)


# initialise FastAPI app
app = FastAPI()


# FastAPI app startup event
@app.on_event("startup")
async def startup_event():
    # run at start up of the FastAPI app
    logger = logging.getLogger(__name__)

    # load config
    global model_assets_dir, data_output_dir
    model_assets_dir = config['model_assets_dir']
    data_output_dir = config['data_output_dir']

    global inference_object
    logger.info('Initialising Inference object')
    inference_object = Inference(model_assets_dir=model_assets_dir,
                                 data_output_dir=data_output_dir)


@app.get("/query")
def run_inference(text: str,
                  #   current_user: Annotated[
                  #      User, Depends(get_current_active_user)]
                  #  # for OAuth2 authentication with bearer token
                  ):

    global inference_object
    if text is None:
        return "No text found, please include a \
?text=blah parameter in the URL", 400
    response = inference_object.run_inference(data=text)
    return str(response[0]), 200


@app.get("/status")
def check_status(
                 #   current_user: Annotated[
                 #      User, Depends(get_current_active_user)]
                 #  # for OAuth2 authentication with bearer token
                 ):

    global inference_object
    return str(inference_object.status), 200


# Takes user_id and password and returns OAuth2 bearer token to be
# put in HTTP request headers
# @app.post("/token")
# async def login_for_access_token(
#     form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
# ) -> Token:
#     user = authenticate_user(fake_users_db,
#                              form_data.username,
#                              form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )
#     return Token(access_token=access_token, token_type="bearer")


# this endpoint is used for pinging the app; not secured by OAuth2
@app.get("/")
def root():
    return "Server running", 200


if __name__ == "__main__":
    port = int(os.getenv('PORT', 3100))
    uvicorn.run("main:app",  # filename of this script
                host="0.0.0.0", port=port,
                reload=False, log_level="debug")
