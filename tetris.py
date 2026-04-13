"""
Streamlit 테트리스 게임
실행: streamlit run tetris.py
"""
import streamlit as st
import random
import time
import json
from pathlib import Path

# ── 페이지 설정 ──
st.set_page_config(page_title="테트리스", layout="wide")

# ── 상수 ──
COLS = 10
ROWS = 20
CELL_SIZE = 28
EMPTY = 0

# 테트로미노 정의 (각 회전 상태)
SHAPES = {
    "I": [[(0,0),(0,1),(0,2),(0,3)], [(0,0),(1,0),(2,0),(3,0)]],
    "O": [[(0,0),(0,1),(1,0),(1,1)]],
    "T": [[(0,0),(0,1),(0,2),(1,1)], [(0,0),(1,0),(2,0),(1,1)],
           [(1,0),(1,1),(1,2),(0,1)], [(0,0),(1,0),(2,0),(1,-1)]],
    "S": [[(0,1),(0,2),(1,0),(1,1)], [(0,0),(1,0),(1,1),(2,1)]],
    "Z": [[(0,0),(0,1),(1,1),(1,2)], [(0,1),(1,0),(1,1),(2,0)]],
    "L": [[(0,0),(0,1),(0,2),(1,0)], [(0,0),(1,0),(2,0),(2,1)],
           [(1,0),(1,1),(1,2),(0,2)], [(0,0),(0,1),(1,1),(2,1)]],
    "J": [[(0,0),(0,1),(0,2),(1,2)], [(0,0),(1,0),(2,0),(0,1)],
           [(0,0),(1,0),(1,1),(1,2)], [(2,0),(0,1),(1,1),(2,1)]],
}

# 블록 색상 팔레트
COLORS = [
    "#FF6B6B", "#FFE66D", "#4ECDC4", "#45B7D1",
    "#96CEB4", "#FFEAA7", "#DDA0DD", "#FF8C00",
    "#00CED1", "#FF69B4", "#7B68EE", "#32CD32",
]

# 배경 그라데이션 색상
BG_GRADIENT = "linear-gradient(135deg, #0c0c3a 0%, #1a0a2e 25%, #16213e 50%, #0f3460 75%, #1a1a5e 100%)"

# 최고 점수 파일
HIGH_SCORE_FILE = Path(__file__).parent / ".tetris_highscore.json"


def load_high_score():
    try:
        if HIGH_SCORE_FILE.exists():
            with open(HIGH_SCORE_FILE, "r") as f:
                return json.load(f).get("high_score", 0)
    except Exception:
        pass
    return 0


def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, "w") as f:
            json.dump({"high_score": score}, f)
    except Exception:
        pass


def init_state():
    """게임 상태 초기화"""
    st.session_state.board = [[EMPTY] * COLS for _ in range(ROWS)]
    st.session_state.score = 0
    st.session_state.stage = 1
    st.session_state.lines_cleared = 0
    st.session_state.high_score = load_high_score()
    st.session_state.game_over = False
    st.session_state.start_time = time.time()
    st.session_state.paused = False
    spawn_piece()


def spawn_piece():
    """새 블록 생성"""
    shape_name = random.choice(list(SHAPES.keys()))
    color = random.choice(COLORS)
    st.session_state.current = {
        "shape": shape_name,
        "rotation": 0,
        "row": 0,
        "col": COLS // 2 - 1,
        "color": color,
    }
    # 다음 블록 미리보기
    if "next_piece" not in st.session_state:
        st.session_state.next_piece = {
            "shape": random.choice(list(SHAPES.keys())),
            "color": random.choice(COLORS),
        }
    # 충돌 체크 — 게임 오버
    cells = get_cells()
    if cells and check_collision(cells):
        st.session_state.game_over = True


def get_cells(piece=None):
    """현재 블록의 셀 좌표 반환"""
    if piece is None:
        piece = st.session_state.get("current")
    if not piece:
        return []
    shape_name = piece["shape"]
    rotation = piece["rotation"] % len(SHAPES[shape_name])
    offsets = SHAPES[shape_name][rotation]
    return [(piece["row"] + dr, piece["col"] + dc) for dr, dc in offsets]


def check_collision(cells):
    """충돌 확인"""
    for r, c in cells:
        if r < 0 or r >= ROWS or c < 0 or c >= COLS:
            return True
        if st.session_state.board[r][c] != EMPTY:
            return True
    return False


def lock_piece():
    """블록 고정"""
    cells = get_cells()
    color = st.session_state.current["color"]
    for r, c in cells:
        if 0 <= r < ROWS and 0 <= c < COLS:
            st.session_state.board[r][c] = color
    clear_lines()
    # 다음 블록을 현재 블록으로
    nxt = st.session_state.next_piece
    st.session_state.current = {
        "shape": nxt["shape"],
        "rotation": 0,
        "row": 0,
        "col": COLS // 2 - 1,
        "color": nxt["color"],
    }
    st.session_state.next_piece = {
        "shape": random.choice(list(SHAPES.keys())),
        "color": random.choice(COLORS),
    }
    # 충돌 → 게임 오버
    if check_collision(get_cells()):
        st.session_state.game_over = True


def clear_lines():
    """완성된 줄 제거"""
    board = st.session_state.board
    new_board = [row for row in board if any(cell == EMPTY for cell in row)]
    cleared = ROWS - len(new_board)
    if cleared > 0:
        for _ in range(cleared):
            new_board.insert(0, [EMPTY] * COLS)
        st.session_state.board = new_board
        # 점수 계산
        points = {1: 100, 2: 300, 3: 500, 4: 800}
        st.session_state.score += points.get(cleared, cleared * 200) * st.session_state.stage
        st.session_state.lines_cleared += cleared
        # 스테이지 업 (10줄마다)
        st.session_state.stage = st.session_state.lines_cleared // 10 + 1
        # 최고 점수 갱신
        if st.session_state.score > st.session_state.high_score:
            st.session_state.high_score = st.session_state.score
            save_high_score(st.session_state.high_score)


def move(dr, dc):
    """블록 이동"""
    if st.session_state.game_over or st.session_state.paused:
        return
    piece = st.session_state.current
    piece["row"] += dr
    piece["col"] += dc
    if check_collision(get_cells()):
        piece["row"] -= dr
        piece["col"] -= dc
        if dr > 0:  # 아래로 이동 중 충돌 → 고정
            lock_piece()


def rotate():
    """블록 회전"""
    if st.session_state.game_over or st.session_state.paused:
        return
    piece = st.session_state.current
    old_rot = piece["rotation"]
    piece["rotation"] = (old_rot + 1) % len(SHAPES[piece["shape"]])
    if check_collision(get_cells()):
        # 벽차기 시도
        for offset in [1, -1, 2, -2]:
            piece["col"] += offset
            if not check_collision(get_cells()):
                return
            piece["col"] -= offset
        piece["rotation"] = old_rot


def hard_drop():
    """하드 드롭"""
    if st.session_state.game_over or st.session_state.paused:
        return
    piece = st.session_state.current
    while True:
        piece["row"] += 1
        if check_collision(get_cells()):
            piece["row"] -= 1
            lock_piece()
            break


def get_ghost_cells():
    """고스트(미리보기) 위치 계산"""
    piece = st.session_state.current.copy()
    piece = dict(piece)
    while True:
        piece["row"] += 1
        cells = get_cells(piece)
        if check_collision(cells):
            piece["row"] -= 1
            return get_cells(piece)


def render_board_html():
    """보드를 HTML로 렌더링"""
    board = [row[:] for row in st.session_state.board]

    # 고스트 블록
    ghost = get_ghost_cells()
    current_cells = get_cells()
    color = st.session_state.current["color"] if st.session_state.current else "#fff"

    ghost_set = set()
    for r, c in ghost:
        if 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == EMPTY:
            ghost_set.add((r, c))

    # 현재 블록
    current_set = {}
    for r, c in current_cells:
        if 0 <= r < ROWS and 0 <= c < COLS:
            current_set[(r, c)] = color

    cells_html = ""
    for r in range(ROWS):
        for c in range(COLS):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            if (r, c) in current_set:
                fill = current_set[(r, c)]
                cells_html += f'<rect x="{x}" y="{y}" width="{CELL_SIZE}" height="{CELL_SIZE}" fill="{fill}" stroke="#ffffff22" rx="3"/>'
                cells_html += f'<rect x="{x+2}" y="{y+2}" width="{CELL_SIZE-4}" height="{CELL_SIZE-4}" fill="{fill}" stroke="#ffffff44" rx="2" opacity="0.8"/>'
            elif board[r][c] != EMPTY:
                fill = board[r][c]
                cells_html += f'<rect x="{x}" y="{y}" width="{CELL_SIZE}" height="{CELL_SIZE}" fill="{fill}" stroke="#ffffff22" rx="3"/>'
                cells_html += f'<rect x="{x+2}" y="{y+2}" width="{CELL_SIZE-4}" height="{CELL_SIZE-4}" fill="{fill}" stroke="#ffffff44" rx="2" opacity="0.7"/>'
            elif (r, c) in ghost_set:
                cells_html += f'<rect x="{x}" y="{y}" width="{CELL_SIZE}" height="{CELL_SIZE}" fill="{color}" stroke="#ffffff22" rx="3" opacity="0.2"/>'
            else:
                cells_html += f'<rect x="{x}" y="{y}" width="{CELL_SIZE}" height="{CELL_SIZE}" fill="#1a1a3e" stroke="#ffffff08" rx="2"/>'

    w = COLS * CELL_SIZE
    h = ROWS * CELL_SIZE
    svg = f'''<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <filter id="glow"><feGaussianBlur stdDeviation="2" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <rect width="{w}" height="{h}" fill="#0a0a2e" rx="8"/>
    {cells_html}
    </svg>'''
    return svg


def render_next_piece_html():
    """다음 블록 미리보기 SVG"""
    nxt = st.session_state.get("next_piece", {})
    if not nxt:
        return ""
    shape_name = nxt["shape"]
    color = nxt["color"]
    offsets = SHAPES[shape_name][0]
    size = 22
    cells_html = ""
    for dr, dc in offsets:
        x = dc * size + size
        y = dr * size + size
        cells_html += f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{color}" stroke="#ffffff33" rx="3"/>'
    return f'''<svg width="{size*6}" height="{size*5}" xmlns="http://www.w3.org/2000/svg">
    <rect width="{size*6}" height="{size*5}" fill="#0a0a2e" rx="6"/>
    {cells_html}</svg>'''


# ── 세션 초기화 ──
if "board" not in st.session_state:
    init_state()

# ── 자동 낙하 처리 ──
if "last_drop" not in st.session_state:
    st.session_state.last_drop = time.time()

# ── CSS 스타일 ──
st.markdown(f"""
<style>
    .stApp {{
        background: {BG_GRADIENT};
    }}
    .game-container {{
        display: flex;
        justify-content: center;
        gap: 20px;
        padding: 10px;
    }}
    .info-panel {{
        background: rgba(10, 10, 46, 0.8);
        border: 2px solid #4ECDC4;
        border-radius: 12px;
        padding: 15px;
        color: white;
        min-width: 160px;
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
    }}
    .info-label {{
        font-size: 12px;
        color: #4ECDC4;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }}
    .info-value {{
        font-size: 28px;
        font-weight: bold;
        color: #FFE66D;
        text-shadow: 0 0 10px rgba(255, 230, 109, 0.5);
    }}
    .title {{
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #FFE66D, #4ECDC4, #45B7D1, #DDA0DD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
        margin-bottom: 10px;
        letter-spacing: 6px;
    }}
    .game-over-text {{
        text-align: center;
        font-size: 36px;
        color: #FF6B6B;
        font-weight: bold;
        text-shadow: 0 0 20px rgba(255, 107, 107, 0.8);
        animation: pulse 1s ease-in-out infinite alternate;
    }}
    @keyframes pulse {{
        from {{ opacity: 0.6; }}
        to {{ opacity: 1.0; }}
    }}
    div[data-testid="stHorizontalBlock"] {{
        gap: 0.5rem;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #4ECDC4, #45B7D1);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        padding: 8px 16px;
        box-shadow: 0 0 10px rgba(78, 205, 196, 0.4);
        transition: all 0.2s;
        width: 100%;
    }}
    .stButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.6);
    }}
</style>
""", unsafe_allow_html=True)

# ── 타이틀 ──
st.markdown('<div class="title">T E T R I S</div>', unsafe_allow_html=True)

# ── 경과 시간 계산 ──
if not st.session_state.game_over and not st.session_state.paused:
    elapsed = int(time.time() - st.session_state.start_time)
else:
    elapsed = int(st.session_state.get("frozen_time", time.time() - st.session_state.start_time))
    if not st.session_state.game_over and st.session_state.paused:
        elapsed = int(st.session_state.get("pause_elapsed", 0))

minutes, seconds = divmod(elapsed, 60)
hours, minutes = divmod(minutes, 60)
time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ── 자동 낙하 ──
drop_interval = max(0.1, 1.0 - (st.session_state.stage - 1) * 0.08)
now = time.time()
if not st.session_state.game_over and not st.session_state.paused:
    if now - st.session_state.last_drop > drop_interval:
        move(1, 0)
        st.session_state.last_drop = now

# ── 레이아웃 ──
left_info, board_col, right_info = st.columns([1.2, 2, 1.2])

with left_info:
    st.markdown(f'''
    <div class="info-panel">
        <div class="info-label">SCORE 점수</div>
        <div class="info-value">{st.session_state.score:,}</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(f'''
    <div class="info-panel">
        <div class="info-label">LINES 라인</div>
        <div class="info-value">{st.session_state.lines_cleared}</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(f'''
    <div class="info-panel">
        <div class="info-label">HIGH SCORE 최고점수</div>
        <div class="info-value" style="color: #FF6B6B;">{st.session_state.high_score:,}</div>
    </div>
    ''', unsafe_allow_html=True)

with board_col:
    if st.session_state.game_over:
        st.markdown('<div class="game-over-text">GAME OVER</div>', unsafe_allow_html=True)

    board_svg = render_board_html()
    st.markdown(f'<div style="display:flex;justify-content:center;">{board_svg}</div>', unsafe_allow_html=True)

    # 조작 버튼
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("◀", key="left"):
            move(0, -1)
            st.rerun()
    with c2:
        if st.button("▶", key="right"):
            move(0, 1)
            st.rerun()
    with c3:
        if st.button("▼", key="down"):
            move(1, 0)
            st.rerun()
    with c4:
        if st.button("⟳", key="rotate"):
            rotate()
            st.rerun()
    with c5:
        if st.button("⏬", key="drop"):
            hard_drop()
            st.rerun()

    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("🔄 새 게임", key="new_game"):
            init_state()
            st.rerun()
    with cc2:
        if st.session_state.game_over:
            pass
        elif st.session_state.paused:
            if st.button("▶ 계속", key="resume"):
                st.session_state.paused = False
                pause_dur = time.time() - st.session_state.pause_start
                st.session_state.start_time += pause_dur
                st.rerun()
        else:
            if st.button("⏸ 일시정지", key="pause"):
                st.session_state.paused = True
                st.session_state.pause_start = time.time()
                st.session_state.pause_elapsed = elapsed
                st.rerun()

with right_info:
    st.markdown(f'''
    <div class="info-panel">
        <div class="info-label">TIME 시간</div>
        <div class="info-value" style="font-size:22px;">{time_str}</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(f'''
    <div class="info-panel">
        <div class="info-label">NEXT 다음 블록</div>
        <div style="display:flex;justify-content:center;margin-top:8px;">
            {render_next_piece_html()}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(f'''
    <div class="info-panel">
        <div class="info-label">STAGE 스테이지</div>
        <div class="info-value" style="color: #96CEB4;">{st.session_state.stage}</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # 속도 표시
    st.markdown(f'''
    <div class="info-panel">
        <div class="info-label">SPEED 속도</div>
        <div class="info-value" style="font-size:20px; color: #DDA0DD;">{1/drop_interval:.1f}x</div>
    </div>
    ''', unsafe_allow_html=True)

# ── 자동 리프레시 (게임 진행 중일 때) ──
if not st.session_state.game_over and not st.session_state.paused:
    time.sleep(drop_interval)
    st.rerun()
