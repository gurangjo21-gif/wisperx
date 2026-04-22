# -*- coding: utf-8 -*-
"""WhisperX + 화자분리 Streamlit 전사 도구."""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import streamlit as st
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

st.set_page_config(page_title="WhisperX 전사", layout="wide")
st.title("WhisperX + 화자분리 전사 도구")


def _default_hf_token() -> str:
    # 프로토타입용 fallback. 운영 전 반드시 Revoke 후 secrets/env 로 옮길 것.
    HARDCODED = "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"
    try:
        token = st.secrets.get("HF_TOKEN", "")
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "") or HARDCODED


_on_cuda = torch.cuda.is_available()
_default_model_index = 0 if _on_cuda else 3  # cuda 면 large-v3, cpu 면 small 을 기본값

if not _on_cuda:
    st.info(
        "현재 **CPU 환경** 입니다. large-v3 모델은 메모리 부족으로 실패하거나 "
        "매우 느릴 수 있습니다. 기본값 `small` 유지 또는 `base` 사용을 권장합니다."
    )

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("설정")

    hf_token = st.text_input(
        "Hugging Face 토큰",
        value=_default_hf_token(),
        type="password",
        help="pyannote 화자분리 모델 접근용. huggingface.co/settings/tokens",
    )

    model_size = st.selectbox(
        "Whisper 모델",
        ["large-v3", "large-v2", "medium", "small", "base"],
        index=_default_model_index,
    )
    language_input = st.text_input(
        "언어 코드 (비우면 자동 감지)", value="ko",
        help="ko, en, ja … 비워두면 Whisper 가 자동 감지. 자동감지 시 Align 모델은 사용 불가.",
    )
    language = language_input.strip() or None
    use_diarization = st.checkbox("화자 분리 사용", value=True)
    diarize_model_name = st.selectbox(
        "화자분리 모델",
        ["pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization-community-1"],
        index=0,
        help="두 모델 모두 huggingface.co 에서 약관 수락 필요 (게이팅)",
    )
    batch_size = st.number_input("배치 크기 (Whisper)", min_value=1, max_value=32, value=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    st.caption(f"device: `{device}` / compute_type: `{compute_type}`")

    if st.button("모델 캐시 비우기"):
        st.cache_resource.clear()
        st.success("캐시를 비웠습니다. 다음 실행 시 모델을 다시 로드합니다.")

    if shutil.which("ffmpeg") is None:
        st.error("ffmpeg 을 PATH 에서 찾지 못했습니다. 설치 후 다시 실행하세요.")


# ---------------- 모델 로드 (캐시) ----------------
@st.cache_resource(show_spinner="Whisper 모델 로드 중...")
def load_whisper(model_size: str, device: str, compute_type: str, language):
    kwargs = {"device": device, "compute_type": compute_type}
    if language:
        kwargs["language"] = language
    return whisperx.load_model(model_size, **kwargs)


@st.cache_resource(show_spinner="Align 모델 로드 중...")
def load_align(language: str, device: str):
    return whisperx.load_align_model(language_code=language, device=device)


@st.cache_resource(show_spinner="화자분리 모델 로드 중...")
def load_diarize(hf_token: str, device: str, model_name: str):
    if not hf_token:
        return None
    try:
        return DiarizationPipeline(
            model_name=model_name, token=hf_token, device=device,
        )
    except Exception as e:
        # huggingface_hub.errors.GatedRepoError 등
        if "Gated" in type(e).__name__ or "401" in str(e) or "gated" in str(e).lower():
            raise RuntimeError(
                f"HF 게이팅 모델 '{model_name}' 접근 거부됨.\n"
                f"브라우저에서 https://huggingface.co/{model_name} 에 로그인하고 "
                f"'Agree and access repository' 를 클릭해 약관을 수락한 뒤 다시 시도하세요."
            ) from e
        raise


# ---------------- 처리 함수 ----------------
def convert_audio(input_path: str) -> str:
    output_path = str(Path(input_path).with_suffix("")) + "_converted.wav"
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", output_path,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg 변환 실패: " + proc.stderr.decode("utf-8", errors="ignore")[-400:]
        )
    return output_path


def _fmt_ts(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(rows: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(rows, 1):
        label = f"[{r['speaker']}] " if r.get("speaker") and r["speaker"] != "UNKNOWN" else ""
        blocks.append(
            f"{i}\n{_fmt_ts(r['start'])} --> {_fmt_ts(r['end'])}\n{label}{r['text']}\n"
        )
    return "\n".join(blocks)


def transcribe_file(
    audio_path: str, model, model_a, metadata, diarize_model,
    device: str, language, batch_size: int,
):
    converted = convert_audio(audio_path)
    try:
        transcribe_kwargs = {"batch_size": batch_size}
        if language:
            transcribe_kwargs["language"] = language
        result = model.transcribe(converted, **transcribe_kwargs)

        if model_a is not None:
            result = whisperx.align(
                result["segments"], model_a, metadata, converted, device,
                return_char_alignments=False,
            )

        if diarize_model is not None:
            diarize_df = diarize_model(converted)
            if isinstance(diarize_df, tuple):
                diarize_df = diarize_df[0]
            if diarize_df is not None and not diarize_df.empty:
                result = whisperx.assign_word_speakers(diarize_df, result)

        lines, rows = [], []
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            start = round(float(seg.get("start", 0.0)), 2)
            end = round(float(seg.get("end", 0.0)), 2)
            text = (seg.get("text") or "").strip()
            lines.append(f"[{speaker}] {start}s ~ {end}s : {text}")
            rows.append({"speaker": speaker, "start": start, "end": end, "text": text})
        return "\n".join(lines), rows
    finally:
        try:
            os.unlink(converted)
        except OSError:
            pass


# ---------------- UI ----------------
uploaded_files = st.file_uploader(
    "오디오 파일 업로드 (다중 선택 가능)",
    type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "webm", "mkv", "aac"],
    accept_multiple_files=True,
)

run = st.button("전사 시작", type="primary", disabled=not uploaded_files)

if run:
    if use_diarization and not hf_token:
        st.error("화자 분리를 사용하려면 사이드바에 Hugging Face 토큰을 입력하세요.")
        st.stop()

    try:
        model = load_whisper(model_size, device, compute_type, language)
        if language:
            model_a, metadata = load_align(language, device)
        else:
            model_a, metadata = None, None
            st.info("언어 자동 감지 모드: 단어 단위 Align 은 건너뜁니다 (세그먼트 시간만 사용).")
        diarize_model = (
            load_diarize(hf_token, device, diarize_model_name) if use_diarization else None
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    progress = st.progress(0.0)
    status = st.empty()
    total = len(uploaded_files)

    for idx, uf in enumerate(uploaded_files, 1):
        status.info(f"[{idx}/{total}] 처리 중: {uf.name}")

        suffix = Path(uf.name).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name

        try:
            t0 = time.time()
            text, rows = transcribe_file(
                tmp_path, model, model_a, metadata, diarize_model,
                device, language, int(batch_size),
            )
            elapsed = time.time() - t0

            with st.expander(f"{uf.name}  ({elapsed:.1f}s)", expanded=True):
                st.text_area("전사 결과", text, height=300, key=f"text_{idx}")
                col1, col2, col3 = st.columns(3)
                col1.download_button(
                    "TXT 다운로드",
                    data=text.encode("utf-8"),
                    file_name=Path(uf.name).stem + ".txt",
                    mime="text/plain",
                    key=f"dl_txt_{idx}",
                )
                col2.download_button(
                    "SRT 다운로드",
                    data=build_srt(rows).encode("utf-8"),
                    file_name=Path(uf.name).stem + ".srt",
                    mime="application/x-subrip",
                    key=f"dl_srt_{idx}",
                )
                col3.download_button(
                    "CSV 다운로드",
                    data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                    file_name=Path(uf.name).stem + ".csv",
                    mime="text/csv",
                    key=f"dl_csv_{idx}",
                )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.error(f"{uf.name} 처리 실패: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        progress.progress(idx / total)

    status.success(f"완료: {total}개 파일")
