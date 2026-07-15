"""
Uploads generated reports to a shared Supabase project for permanent, searchable
storage. Copied identically into each app's repo — no shared package between apps.
Never blocks or breaks the app's existing local download flow: every failure path
returns None instead of raising.
"""

import base64
import io
import logging
import math
import struct
import time
import wave

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
        # returning="minimal" is required: the anon key can only INSERT, not
        # SELECT, and Postgres/PostgREST's default behavior tries to read the
        # row back after inserting (needs SELECT). Without this, every insert
        # is rejected by RLS even though the INSERT policy itself is correct.
        client.table("reports").insert({
            "app_name": app_name,
            "module_name": module_name,
            "user_prompt": user_prompt,
            "interim_steps": interim_steps,
            "final_output_summary": final_output_summary,
            "file_url": file_url,
            "file_type": file_type,
            "status": status,
        }, returning="minimal").execute()
    except Exception as e:
        logger.warning(f"archive_helper: file uploaded but metadata row failed: {e}")

    return file_url


def notify_archived():
    """
    Shows a bold on-screen confirmation plus a short chime after a successful
    archive, so the user isn't left guessing whether anything happened. Call
    this only when save_report() returned a real URL, not None.
    """
    import streamlit as st

    st.success("✅ **Report archived successfully** — a permanent copy has been saved to cloud storage.")
    audio_b64 = _generate_chime_wav_base64()
    st.markdown(
        f'<audio autoplay="true"><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


def _generate_chime_wav_base64():
    """Synthesizes a short two-note confirmation chime — no external audio file needed."""
    sample_rate = 44100
    notes = [(880.0, 0.12), (1174.0, 0.18)]  # A5 then D6 — a bright, quick "done" sound
    frames = bytearray()
    for freq, duration in notes:
        n_samples = int(sample_rate * duration)
        fade_samples = max(1, int(sample_rate * 0.01))
        for i in range(n_samples):
            t = i / sample_rate
            fade = min(i / fade_samples, 1.0, (n_samples - i) / fade_samples)
            sample = math.sin(2 * math.pi * freq * t) * fade * 0.3
            frames += struct.pack("<h", int(sample * 32767))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
            # No upsert: the anon key is scoped INSERT-only via RLS (see the
            # Report Archive setup guide). Requesting upsert:true made Supabase
            # require UPDATE permission too, which this key doesn't have --
            # confirmed directly: the exact same call succeeds without upsert
            # and fails with "new row violates row-level security policy" the
            # instant upsert is added, even on a brand-new path that has never
            # been uploaded before. A real duplicate-path collision (e.g.
            # running the same topic twice) is handled by the idempotency
            # guard in ui.py instead, not by this flag.
            client.storage.from_(app_name).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": content_type},
            )
            return client.storage.from_(app_name).get_public_url(storage_path)
        except Exception as e:
            logger.warning(f"archive_helper: upload attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_PAUSE_SECONDS)

    return None
