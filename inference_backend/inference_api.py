# reference: https://fastapi.tiangolo.com/tutorial/security/

import logging
import os

import uvicorn

# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from pydantic import BaseModel
from utils.blob_utils import (
    delete_blob,
    download_blob,
    download_container,
    save_file_to_blob,
)
from utils.chatbot_classes import ChatEngine, KnowledgeBase, UserChatHistory
from utils.service_context_utils import load_api_configs

from fastapi import FastAPI, HTTPException  # Depends,; status

# from datetime import datetime, timedelta, timezone
# from typing import Annotated


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# secrets from env vars (.env)
# openai token is already in env var
connection_string = os.environ['AZURE_CONNECTION_STRING']

# export USER_ID=<user_id> before deploying the chat engine container
user_id_given = os.environ['USER_ID']


# downolad app configs from Azure blob and load configs
global configs
WORK_DIR = '/code'
API_CONFIG_CONTAINER = 'chat-engine-api-config'
API_CONFIG_BLOB_NAME = 'api_config.yml'
API_CONFIG_DIR = 'configs'
download_blob(
        connection_string=connection_string,
        container=API_CONFIG_CONTAINER,
        blob_name=API_CONFIG_BLOB_NAME,
        work_dir=WORK_DIR,
        destination_dir=API_CONFIG_DIR,)

configs = load_api_configs(os.path.join(WORK_DIR,
                                        API_CONFIG_DIR,
                                        API_CONFIG_BLOB_NAME))
configs['work_dir'] = WORK_DIR
common_volume_dir = os.path.join(configs['work_dir'],
                                 configs['common_subdir'])
user_cache_dir = os.path.join(configs['work_dir'],
                              configs['user_cache_subdir'])


LLAMA_INDEX_CACHE_DIR = os.path.join(
    common_volume_dir,
    '.cache', 'llama_index'
)

TRANSFORMERS_CACHE_DIR = os.path.join(
    common_volume_dir,
    '.cache', 'huggingface/hub'
)


# set llama_index_cache_path; default is /tmp/llama_index
# local embedding models are saved here
os.makedirs(LLAMA_INDEX_CACHE_DIR, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)
os.environ['LLAMA_INDEX_CACHE_DIR'] = LLAMA_INDEX_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR


configs.update({'index_dir': os.path.join(common_volume_dir,
                                          configs['index_dir']),
                'chat_history_dir': os.path.join(user_cache_dir,
                                                 configs['chat_history_dir'])
                })
configs.update({'work_dir': ''})


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


# initialise FastAPI app
app = FastAPI()


# FastAPI app startup event
@app.on_event("startup")
async def startup_event():
    # run at start up of the FastAPI app
    logger = logging.getLogger(__name__)

    global configs
    if not configs['use_weaviate']:
        index_store_container = configs.pop(
            'index_store_container')
        logger.info('Downloading knowledge base index from blob')
        os.path.join(configs['index_dir'],
                                  configs['dataset_subdir'])
        download_container(
            connection_string=connection_string,
            container=index_store_container,
            work_dir=configs['work_dir'],
            destination_data_dir=configs['index_dir'],
            dataset_subdir=configs['dataset_subdir'])

    global index
    logger.info('Loading knowledge base')
    knowledge_base = KnowledgeBase(**configs)
    index = knowledge_base.index

    global chat_history, chat_history_container, user_chat_history_blob_name
    logger.info('Downloading user chat history from blob')
    chat_history_container = configs.pop(
        'chat_history_container')
    user_chat_history_blob_name = str(user_id_given) + '_chat_history.json'

    download_blob(
        connection_string=connection_string,
        container=chat_history_container,
        blob_name=user_chat_history_blob_name,
        work_dir=configs['work_dir'],
        destination_dir=configs['chat_history_dir'],)
    logger.info('Loading user chat history')
    chat_history = UserChatHistory(
        user_id=user_id_given,
        chat_history_dir=configs['chat_history_dir'],
        chat_history_filename=user_chat_history_blob_name)
    chat_history.load_chat_history()

    global chat_engine
    logger.info('Initialising chat engine')
    chat_engine = ChatEngine(
            index=index,
            engine_type=configs['chat_engine_type'],
            custom_chat_history=chat_history.chat_history_llamaindex,
            query_engine_args=configs['query_engine_args'],
            chat_engine_args=configs['chat_engine_args'])


@app.get("/{user_id}/query")
def query_chat_engine(user_id: str,
                      text: str,
                      #   current_user: Annotated[
                      #      User, Depends(get_current_active_user)]
                      #  # for OAuth2 authentication with bearer token
                      ):
    if user_id_given != user_id:
        raise HTTPException(
            status_code=400,
            detail=f'entered user_id {user_id} does not correspond to host')

    global chat_engine, chat_history
    query_text = text
    if query_text is None:
        return "No text found, please include a \
?text=blah parameter in the URL", 400
    chat_history.append_chat_history(role='user', content=query_text)
    response = chat_engine.get_response(query_text)
    chat_history.append_chat_history(role='assistant',
                                     content=response.response)
    chat_history.save_chat_history()
    save_file_to_blob(connection_string=connection_string,
                      container=chat_history_container,
                      blob_name=user_chat_history_blob_name,
                      filepath=os.path.join(
                          chat_history._chat_history_dir,
                          user_chat_history_blob_name))
    return str(response.response), 200


@app.get("/{user_id}/chat_history")
def get_chat_history(user_id: str,
                     #  current_user: Annotated[
                     #      User, Depends(get_current_active_user)]
                     #  # for OAuth2 authentication with bearer token
                     ):
    if user_id_given != user_id:
        raise HTTPException(
            status_code=400,
            detail=f'entered user_id {user_id} does not correspond to host')

    global chat_history
    response = chat_history.display_chat_history()
    if response is None:
        response = []
    return response, 200


@app.put("/{user_id}/clear_chat_history")
def clear_chat_history(user_id: str,
                       #    current_user: Annotated[
                       #      User, Depends(get_current_active_user)]
                       #  # for OAuth2 authentication with bearer token
                       ):
    if user_id_given != user_id:
        raise HTTPException(
            status_code=400,
            detail=f'entered user_id {user_id} does not correspond to host')

    global chat_engine, chat_history, configs
    chat_history.clear_chat_history()
    delete_blob(connection_string=connection_string,
                container=chat_history_container,
                blob_name=user_chat_history_blob_name)
    # re-initialise the chat engine
    chat_history = UserChatHistory(
        user_id=user_id_given,
        chat_history_dir=configs['chat_history_dir'],
        chat_history_filename=user_chat_history_blob_name)
    chat_engine = ChatEngine(
        index=index,
        engine_type=configs['chat_engine_type'],
        custom_chat_history=chat_history.chat_history_llamaindex,
        query_engine_args=configs['query_engine_args'],
        chat_engine_args=configs['chat_engine_args'])
    return 'Chat history cleared', 200


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
    port = configs['api_port']
    uvicorn.run("main:app",  # filename of this script
                host="0.0.0.0", port=port,
                reload=False, log_level="debug")