# WhisperX 전사 + 화자분리 (Streamlit)

오디오 파일을 업로드하면 WhisperX 로 전사하고 pyannote 로 화자분리한 결과를
TXT / SRT / CSV 로 내려주는 Streamlit 앱.

---

## 빠른 배포 (Streamlit Community Cloud)

1. **Hugging Face 토큰 발급**
   - https://huggingface.co/settings/tokens 에서 Read 권한 토큰 생성
   - 아래 두 모델 페이지에서 약관 수락 (게이팅 모델)
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0

2. **GitHub 에 push** (예: `whisperx-streamlit` 라는 이름의 public repo 생성 후)
   ```bash
   git remote add origin https://github.com/<본인계정>/whisperx-streamlit.git
   git branch -M main
   git push -u origin main
   ```

3. **Streamlit Community Cloud 에서 앱 생성**
   - https://share.streamlit.io → **New app** → 위 repo 선택, branch `main`, file `app.py`
   - **App settings → Secrets** 에 다음 한 줄 붙여넣기:
     ```toml
     HF_TOKEN = "hf_본인토큰"
     ```
   - Deploy 클릭. 첫 빌드는 의존성(torch, whisperx 등) 설치로 5~10 분 정도 소요.

4. 끝. URL 이 발급되며 사이드바에서 모델/언어/화자분리 옵션 조절 가능.

---

## ⚠️ 무료 플랜 한계 (꼭 읽어보세요)

Streamlit Community Cloud 무료 플랜은 **CPU 전용 + 약 2.7GB RAM** 입니다.

| 모델 | CPU 에서 권장도 | 메모리 |
|---|---|---|
| `large-v3` / `large-v2` | ❌ 거의 OOM 또는 매우 느림 | ~3GB+ |
| `medium` | △ 짧은 오디오만 | ~1.5GB |
| `small` (기본) | ✅ 일반 용도 | ~500MB |
| `base` | ✅ 가장 빠름, 정확도 낮음 | ~200MB |

- pyannote 화자분리도 RAM 을 추가로 잡아먹습니다.
- 1 시간짜리 오디오는 무료 플랜에서 비현실적입니다. **5~10 분 단위로 잘라서** 올리세요.
- 진지하게 쓰려면 GPU 가 있는 환경(로컬, Colab, HuggingFace Spaces GPU, Streamlit for Teams 등)으로 옮기세요.

---

## 로컬 실행

### Windows
```cmd
run.bat
```
처음 한 번만 의존성 설치. 이후로는 즉시 실행.
ffmpeg 는 winget 으로 깔린 것을 자동 PATH 에 추가합니다.

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg 가 PATH 에 있어야 합니다 (apt install ffmpeg / brew install ffmpeg)
streamlit run app.py
```

### 로컬에서 HF 토큰 등록 (선택)
`.streamlit/secrets.toml.example` 을 `.streamlit/secrets.toml` 로 복사하고 토큰을 적으면
사이드바에 자동으로 채워집니다. (이 파일은 `.gitignore` 로 보호됨)

---

## 파일 구조

```
.
├── app.py                       # Streamlit 메인
├── requirements.txt             # Python 의존성
├── packages.txt                 # apt 패키지 (ffmpeg 등) — Streamlit Cloud 가 읽음
├── run.bat                      # Windows 로컬 실행 스크립트
├── .streamlit/
│   ├── config.toml              # 업로드 한도 등
│   └── secrets.toml.example     # 시크릿 템플릿
├── .gitignore
└── 위스퍼엑스_모델_프로모션_모델.py   # 원본 Colab 노트북 (참고용, 로컬 실행 X)
```

---

## 보안 주의

- **HF 토큰을 코드에 직접 적지 마세요.** 항상 `secrets.toml` / 환경변수 / Streamlit Cloud Secrets 사용.
- 만약 토큰이 노출됐다면 https://huggingface.co/settings/tokens 에서 즉시 **Revoke** 후 재발급.
