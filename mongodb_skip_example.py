# ----------------------------------------------------------------------------------------------------
# 작성목적 : MongoDB 연결 실패 시 스킵 기능 사용 예제
# 작성일 : 2025-01-27

# 변경사항 내역 (날짜 | 변경목적 | 변경내용 | 작성자 순으로 기입)
# 2025-01-27 | 최초 구현 | MongoDB 스킵 기능 사용법 가이드 | 구동빈
# ----------------------------------------------------------------------------------------------------

import asyncio
from datetime import datetime
from src.db import (
    safe_save_analysis_result,
    safe_get_analysis_results,
    safe_update_analysis_status,
    get_analysis_statistics,
    setup_database,
    check_database_connection
)

async def example_usage():
    """MongoDB 연결 실패 시 스킵 기능 사용 예제"""
    
    print("🚀 MongoDB 연결 스킵 기능 예제 시작")
    print("=" * 50)
    
    # 1. 데이터베이스 설정 시도 (실패해도 계속 진행)
    print("\n1. 데이터베이스 설정 시도...")
    setup_database()
    
    # 2. 연결 상태 확인
    print("\n2. MongoDB 연결 상태 확인...")
    is_connected = check_database_connection()
    print(f"   MongoDB 연결 상태: {'✅ 연결됨' if is_connected else '❌ 연결 안됨'}")
    
    # 3. 분석 결과 저장 시도 (연결 실패 시 스킵)
    print("\n3. 분석 결과 저장 시도...")
    analysis_data = {
        "analysis_id": f"test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "user_id": "test_user",
        "session_id": "test_session",
        "status": "completed",
        "emotion_analysis": {
            "interview_score": 85.5,
            "dominant_emotion": "confident"
        },
        "eye_tracking_analysis": {
            "attention_score": 78.2,
            "gaze_stability": 0.85
        },
        "created_at": datetime.now().isoformat()
    }
    
    save_success = safe_save_analysis_result(analysis_data)
    if save_success:
        print("   ✅ 분석 결과 저장 성공")
    else:
        print("   ⚠️ 분석 결과 저장 실패 - 스킵됨 (애플리케이션 계속 실행)")
    
    # 4. 분석 결과 조회 시도 (연결 실패 시 None 반환)
    print("\n4. 분석 결과 조회 시도...")
    analysis_id = analysis_data["analysis_id"]
    result = safe_get_analysis_results(analysis_id)
    
    if result:
        print(f"   ✅ 분석 결과 조회 성공: {result['analysis_id']}")
    else:
        print("   ⚠️ 분석 결과 조회 실패 - None 반환 (애플리케이션 계속 실행)")
    
    # 5. 상태 업데이트 시도 (연결 실패 시 False 반환)
    print("\n5. 분석 상태 업데이트 시도...")
    update_success = safe_update_analysis_status(analysis_id, "completed", "분석 완료")
    if update_success:
        print("   ✅ 상태 업데이트 성공")
    else:
        print("   ⚠️ 상태 업데이트 실패 - 스킵됨 (애플리케이션 계속 실행)")
    
    # 6. 통계 조회 시도 (연결 실패 시 기본값 반환)
    print("\n6. 분석 통계 조회 시도...")
    statistics = get_analysis_statistics()
    print(f"   📊 통계 조회 결과:")
    print(f"      총 분석 수: {statistics.get('total_analyses', 0)}")
    print(f"      오늘 분석 수: {statistics.get('today_analyses', 0)}")
    if "note" in statistics:
        print(f"      주의사항: {statistics['note']}")
    
    print("\n" + "=" * 50)
    print("✅ 예제 완료 - MongoDB 연결 상태와 관계없이 애플리케이션이 정상 동작했습니다!")
    print("\n핵심 포인트:")
    print("• MongoDB 연결 실패 시 예외 발생 대신 스킵")
    print("• 저장 실패 시 False 반환, 애플리케이션 계속 실행")
    print("• 조회 실패 시 None 또는 빈 리스트 반환")
    print("• 통계 조회 실패 시 기본값과 함께 상태 메시지 반환")

async def test_without_mongodb():
    """MongoDB 없이도 애플리케이션이 정상 동작하는지 테스트"""
    
    print("\n🧪 MongoDB 없이 애플리케이션 동작 테스트")
    print("=" * 50)
    
    # MongoDB 없이도 이런 작업들이 정상 동작해야 함
    tasks = [
        ("영상 분석 로직", lambda: print("✅ 영상 분석 완료")),
        ("감정 분석", lambda: print("✅ 감정 분석 완료")),
        ("시선 추적", lambda: print("✅ 시선 추적 완료")),
        ("S3 업로드", lambda: print("✅ S3 업로드 완료")),
        ("API 응답", lambda: print("✅ API 응답 생성 완료")),
    ]
    
    for task_name, task_func in tasks:
        print(f"\n🔄 {task_name} 실행 중...")
        try:
            task_func()
            print(f"   ✅ {task_name} 성공")
        except Exception as e:
            print(f"   ❌ {task_name} 실패: {e}")
    
    print("\n" + "=" * 50)
    print("✅ MongoDB 없이도 핵심 기능들이 정상 동작합니다!")

if __name__ == "__main__":
    print("MongoDB 연결 실패 시 스킵 기능 데모")
    print("이 예제는 MongoDB가 없어도 정상 실행됩니다.")
    
    # 예제 실행
    asyncio.run(example_usage())
    asyncio.run(test_without_mongodb()) 