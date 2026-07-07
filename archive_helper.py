"""
Uploads generated reports to a shared Supabase project for permanent, searchable
storage. Copied identically into each app's repo — no shared package between apps.
Never blocks or breaks the app's existing local download flow: every failure path
returns None instead of raising.
"""

import logging
import time

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 3
ATTEMPT_TIMEOUT_SECONDS = 10
RETRY_PAUSE_SECONDS = 1

CONTENT_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "png": "image/png",
    "pdf": "application/pdf",
    "csv": "text/csv",
}


def save_report(
    app_name: str,
    module_name: str,
    file_bytes: bytes,
    file_name: str,
    file_type: str,
    user_prompt: str = None,
    interim_steps: dict = None,
    final_output_summary: str = None,
    status: str = "complete",
):
    """
    Uploads file_bytes to Supabase storage under {app_name}/{module_name}/{file_name}
    and logs a metadata row in the `reports` table. Returns the file's public URL on
    success, or None if archiving failed for any reason. Retries the upload up to
    RETRY_ATTEMPTS times before giving up.
    """
    try:
        client = _get_client()
    except Exception as e:
        logger.warning(f"archive_helper: could not set up Supabase client, skipping archive: {e}")
        return None

    file_url = _upload_with_retries(client, app_name, module_name, file_bytes, file_name, file_type)
    if file_url is None:
        return None

    try:
        client.table("reports").insert({
            "app_name": app_name,
            "module_name": module_name,
            "user_prompt": user_prompt,
            "interim_steps": interim_steps,
            "final_output_summary": final_output_summary,
            "file_url": file_url,
            "file_type": file_type,
            "status": status,
        }).execute()
    except Exception as e:
        logger.warning(f"archive_helper: file uploaded but metadata row failed: {e}")

    return file_url


def _get_client():
    import streamlit as st
    from supabase import create_client, ClientOptions

    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]

    return create_client(
        supabase_url,
        supabase_key,
        options=ClientOptions(
            postgrest_client_timeout=ATTEMPT_TIMEOUT_SECONDS,
            storage_client_timeout=ATTEMPT_TIMEOUT_SECONDS,
        ),
    )


def _upload_with_retries(client, app_name, module_name, file_bytes, file_name, file_type):
    storage_path = f"{module_name}/{file_name}"
    content_type = CONTENT_TYPES.get(file_type.lower(), "application/octet-stream")

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            client.storage.from_(app_name).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            return client.storage.from_(app_name).get_public_url(storage_path)
        except Exception as e:
            logger.warning(f"archive_helper: upload attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_PAUSE_SECONDS)

    return None
