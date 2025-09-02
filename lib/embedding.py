from tqdm import tqdm
import openai
import lib.utils

MAX_TRIES = 6

class OpenAIEmbeddings:
    
    _client: openai.OpenAI = None
    @property
    def client(self):
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client
    
    _a_client: openai.AsyncOpenAI = None
    @property
    def a_client(self):
        if self._a_client is None:
            self._a_client = openai.AsyncOpenAI()
        return self._a_client

    def __init__(self, embedding_model: str = 'text-embedding-ada-002'):
        self.embedding_model = embedding_model

    def get_embedding(self, text: str) -> list[float]:
        text_cleaned = self._clean_text_(text)
        return self._call_client_([text_cleaned]).data[0].embedding
    
    async def async_get_embedding(self, text: str) -> list[float]:
        text_cleaned = self._clean_text_(text)
        return (await self._async_call_client_([text_cleaned])).data[0].embedding

    def get_embeddings(
            self, 
            texts: list[str], 
            batch_size: int = 100,
            verbose: bool = False
        ) -> list[list[float]]:
        texts_cleaned = [self._clean_text_(text) for text in texts]
        embeddings:list[list[float]] = []

        if verbose:
            iterator = tqdm(range(0, len(texts_cleaned), batch_size))
        else:
            iterator = range(0, len(texts_cleaned), batch_size)

        for i in iterator:
            response = self._call_client_(texts_cleaned[i:i+batch_size])
            embeddings += [emb.embedding for emb in response.data]
        return embeddings

    async def async_get_embeddings(
            self, 
            texts: list[str], 
            batch_size: int = 100,
            verbose: bool = False
        ) -> list[list[float]]:
        texts_cleaned = [self._clean_text_(text) for text in texts]
        embeddings:list[list[float]] = []

        if verbose:
            iterator = tqdm(range(0, len(texts_cleaned), batch_size))
        else:
            iterator = range(0, len(texts_cleaned), batch_size)

        for i in iterator:
            response = await self._async_call_client_(texts_cleaned[i:i+batch_size])
            embeddings += [emb.embedding for emb in response.data]
        return embeddings

    def _call_client_(self, text: str | list[str]):
        try_count = 0

        while try_count < MAX_TRIES:
            try_count += 1
            try:
                return self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model,
                    timeout=10
                )
            except Exception as e:
                lib.utils.openai_exception_handler(
                    e, 
                    try_count,
                    class_name=self.__class__.__name__,
                    max_tries=MAX_TRIES
                )
                lib.utils.sleep(try_count)

    async def _async_call_client_(self, text: str | list[str]):
        try_count = 0

        while try_count < MAX_TRIES:
            try_count += 1
            try:
                return await self.a_client.embeddings.create(
                    input=text,
                    model=self.embedding_model,
                    timeout=10
                )
            except Exception as e:
                await lib.utils.openai_exception_handler(
                    e, 
                    try_count,
                    class_name=self.__class__.__name__,
                    max_tries=MAX_TRIES
                )
                await lib.utils.wait(try_count)

    def _clean_text_(self, text: str) -> str:
        text = str(text)
        if (text is None) or (text == ""):
            return " "
        return text.strip().replace("\n", " ")
