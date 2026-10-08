from fastapi import FastAPI

app = FastAPI(title="Research Intelligence API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
