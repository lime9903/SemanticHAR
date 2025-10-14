# Git LFS 설정 가이드

checkpoint 파일들이 매우 크기 때문에 (423MB+) Git LFS를 사용해야 합니다.

## 1. Git LFS 설치

### Ubuntu/Debian
```bash
sudo apt-get install git-lfs
```

### macOS
```bash
brew install git-lfs
```

### Windows
https://git-lfs.github.com/ 에서 다운로드

## 2. Git LFS 초기화

```bash
# 저장소에서 Git LFS 설정
git lfs install

# LFS로 관리할 파일 확인
git lfs track "*.pth"
git lfs track "*.pt"

# .gitattributes 확인
cat .gitattributes
```

## 3. Checkpoint 파일 추가

```bash
# 변경사항 스테이징
git add .gitattributes
git add .gitignore
git add checkpoints/

# 커밋
git commit -m "Add model checkpoints with Git LFS"

# 푸시
git push origin main
```

## 4. 다른 사람이 다운로드 받을 때

```bash
# 저장소 클론 (자동으로 LFS 파일도 다운로드)
git clone <repository-url>

# 또는 기존 저장소에서 LFS 파일 가져오기
git lfs pull
```

## 주의사항

- Git LFS는 저장 용량 제한이 있습니다 (GitHub 무료: 1GB storage, 1GB bandwidth/month)
- 대용량 모델 파일은 Hugging Face Hub나 다른 모델 저장소 사용을 권장합니다
- 현재 checkpoint 파일 크기: ~423MB

## 대안: Hugging Face Hub 사용

대용량 모델을 공유할 때 권장하는 방법:

```bash
pip install huggingface_hub

# 모델 업로드
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/text_encoder_trained.pth",
    path_in_repo="text_encoder_trained.pth",
    repo_id="your-username/semantic-har",
    repo_type="model",
)
```

