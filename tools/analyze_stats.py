import pandas as pd

df = pd.read_csv('all_demos.csv')

print('=== 1. 기본 현황 ===')
print(f'총 수집 시연: {len(df)}건')
qual_count = df['is_qualified'].sum()
print(f'공식 인정 (is_qualified=True): {qual_count}건 ({qual_count/len(df)*100:.1f}%)')
print(f'고유 참여자(닉네임): {df["nickname"].nunique()}명')

print('\n=== 2. 도안별(template_name) 분포 ===')
tpl_stat = df.groupby('template_name').agg(
    total=('id', 'count'),
    qualified=('is_qualified', 'sum'),
    qual_rate=('is_qualified', lambda x: f'{x.mean()*100:.1f}%'),
    avg_score=('score', 'mean'),
    avg_error=('avg_error', 'mean'),
    avg_duration=('duration_sec', 'mean')
).sort_values(by='total', ascending=False)
print(tpl_stat)

print('\n=== 3. 입력 기기(input_type) 분포 ===')
print(df['input_type'].value_counts())
print(df.groupby('input_type')['is_qualified'].mean())

print('\n=== 4. 참여자 기여도 상위 8명 (쏠림 현상) ===')
top_users = df['nickname'].value_counts().head(8)
for nick, cnt in top_users.items():
    print(f'  {nick}: {cnt}건 ({cnt/len(df)*100:.1f}%)')

print('\n=== 5. 티어별 분포 ===')
print(df['tier'].value_counts())

print('\n=== 6. 소요 시간 통계 (초) ===')
print(df['duration_sec'].describe())
