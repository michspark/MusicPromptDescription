import csv
import time
import os
import re
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    # 이미지 차단으로 속도 최적화
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=chrome_options)

def extract_duration_refined(html_source):
    """
    이스케이프된 JSON 데이터까지 모두 탐색하는 정규표현식
    패턴 1: \"duration\":315.8 (스크립트 내 이스케이프 형태)
    패턴 2: "duration":315.8 (일반 형태)
    """
    try:
        # 역슬래시가 있거나 없는 duration 패턴을 모두 찾습니다.
        match = re.search(r'\\?"duration\\?":\s*(\d+\.?\d*)', html_source)
        if match:
            total_seconds = float(match.group(1))
            if total_seconds > 0:
                mins = int(total_seconds // 60)
                secs = int(total_seconds % 60)
                return f"{mins}:{secs:02d}"
    except Exception as e:
        pass
    return "N/A"

def scrape_audio_page(driver, url: str) -> Dict[str, str]:
    result = {'url': url, 'title': 'N/A', 'duration': 'N/A'}
    try:
        driver.get(url)

        # 1. 페이지 로딩 대기 (제목 태그가 나타날 때까지)
        wait = WebDriverWait(driver, 15)
        try:
            title_el = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
            result['title'] = title_el.text.strip()
        except:
            pass

        # 2. 아주 짧은 대기 (스크립트 데이터가 브라우저 메모리에 안착할 시간)
        time.sleep(1)

        # 3. 소스코드 전체에서 정밀 추출
        result['duration'] = extract_duration_refined(driver.page_source)

    except Exception as e:
        print(f"  ⚠ 오류 발생: {e}")

    return result

def main():
    input_file = r'C:\Users\MICHA\Codes\PromptDescriptionPrj\data\udio_urls.csv'      # 입력 URL 파일명
    output_file = r'C:\Users\MICHA\Codes\PromptDescriptionPrj\data\udio_durations.csv'  # 저장 결과 파일명

    driver = setup_driver()

    # URL 로드
    urls = []
    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and 'http' in row[0]: urls.append(row[0])

    # 중복 체크
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader: processed.add(row['url'])

    to_process = [u for u in urls if u not in processed]
    fieldnames = ['url', 'title', 'duration']

    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"시작: {len(to_process)}개의 작업을 처리합니다.")

    try:
        for i, url in enumerate(to_process, 1):
            res = scrape_audio_page(driver, url)

            # 파일에 즉시 기록
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows([res])

            print(f"[{i}/{len(to_process)}] {res['title']} | {res['duration']}")

            # 서버 차단 방지용 미세 대기
            time.sleep(0.2)

    finally:
        driver.quit()
        print("작업 완료.")

if __name__ == "__main__":
    main()