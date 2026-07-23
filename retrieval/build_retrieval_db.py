#!/usr/bin/env python3
"""
build_retrieval_db.py (v2) - 基于 device_corpus.jsonl 的885个真实设备ID构建检索索引

从 device_corpus.jsonl 提取所有唯一条目，用规则+设备类型生成中文功能描述。
输出: SQLite FTS5 索引 + JSONL 字典文件
"""
import json, os, sys, sqlite3, argparse, re

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(BASE, 'device_corpus.jsonl')
DEFAULT_OUT = os.path.join(BASE, 'devices_full.jsonl')
DEFAULT_DB = os.path.join(BASE, 'devices_full.db')

# 英文指令关键词 -> 中文动作
VERB_MAP = {
    'access': '访问', 'activate': '激活', 'adjust': '调节', 'align': '对准',
    'analyse': '分析', 'apply': '施加', 'bleed': '排放', 'check': '检查',
    'clean': '清洁', 'close': '关闭', 'connect': '连接', 'deactivate': '停用',
    'disconnect': '断开', 'drain': '排放', 'enable': '启用', 'extract': '提取',
    'flush': '冲洗', 'heating': '加热', 'inspect': '检查', 'install': '安装',
    'isolate': '隔离', 'lock': '锁定', 'lubricate': '润滑', 'maintain': '维护',
    'measure': '测量', 'monitor': '监控', 'open': '打开', 'operate': '操作',
    'position': '定位', 'pressurize': '加压', 'program': '编程', 'pull': '拉',
    'purge': '吹除', 'push': '推', 'read': '读取', 'record': '记录',
    'refuel': '加油', 'release': '释放', 'remove': '拆卸', 'replace': '更换',
    'repair': '修理', 'reset': '重置', 'rig': '安装测试工具', 'rotate': '旋转',
    'run': '运行', 'set': '设置', 'simulate': '模拟', 'supply': '供电',
    'test': '测试', 'tighten': '拧紧', 'torque': '力矩', 'turn': '转动',
    'verify': '验证', 'visual': '目视', 'weigh': '称重', 'zero': '归零',
}

NOUN_MAP = {
    'voltage': '电压', 'current': '电流', 'resistance': '电阻', 'frequency': '频率',
    'torque': '力矩', 'pressure': '压力', 'temperature': '温度', 'force': '力',
    'load': '载荷', 'gap': '间隙', 'leak': '泄漏', 'flow': '流量',
    'speed': '速度', 'position': '位置', 'angle': '角度', 'displacement': '位移',
    'noise': '噪音', 'vibration': '振动', 'oil': '滑油', 'fuel': '燃油',
    'continuity': '通断', 'insulation': '绝缘', 'capacity': '容量',
    'function': '功能', 'system': '系统', 'valve': '活门', 'switch': '开关',
    'sensor': '传感器', 'monitor': '监控器', 'indicator': '指示器',
    'light': '灯光', 'timer': '计时器', 'crimp': '压接', 'seal': '密封',
    'fastener': '紧固件', 'bolt': '螺栓', 'nut': '螺母', 'wire': '导线',
    'connector': '连接器', 'breaker': '跳开关', 'relay': '继电器',
    'actuator': '作动器', 'motor': '马达', 'pump': '泵', 'filter': '滤芯',
    'view': '视野', 'power': '电源', 'ground': '接地', 'signal': '信号',
    'data': '数据', 'and': '并', 'weight': '重量', 'inspect': '检查',
    'ac': '交流', 'dc': '直流', 'ulb': '水下定位信标',
    'io': '输入输出', 'additional': '额外', 'roll': '滚转', 'pitch': '俯仰', 'yaw': '偏航',
    'axial': '轴向', 'radial': '径向', 'lateral': '横向', 'vertical': '纵向',
    'forward': '前向', 'reverse': '反向', 'continuous': '连续',
    'max': '最大', 'min': '最小', 'init': '初始', 'final': '最终',
    'cw': '顺时针', 'ccw': '逆时针', 'multi': '多次',
    'functional': '功能', 'prox': '接近', 'sensor': '传感器',
    'roller': '滚轮', 'heating': '加热', 'lab': '实验室',
}

def command_id_to_chinese(cid):
    """将英文指令ID翻译为中文功能描述"""
    parts = cid.split('_')
    chinese_parts = []
    for p in parts:
        if p in VERB_MAP:
            chinese_parts.append(VERB_MAP[p])
        elif p in NOUN_MAP:
            chinese_parts.append(NOUN_MAP[p])
        elif p.replace('-','').isdigit():
            chinese_parts.append(p)  # Keep numbers
        else:
            # Try to match partial
            matched = False
            for k, v in NOUN_MAP.items():
                if k in p:
                    chinese_parts.append(v)
                    matched = True
                    break
            if not matched:
                chinese_parts.append(p)
    return ''.join(chinese_parts) if chinese_parts else cid

def build_desc(cid, dtype):
    """生成完整的中文功能说明"""
    action_desc = command_id_to_chinese(cid)
    if dtype:
        return f'{dtype}: {action_desc}'
    return action_desc

def build_index(corpus_path, out_jsonl, out_db):
    # Load unique devices from corpus
    id_to_info = {}
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            did = str(r.get('设备指令号', '')).strip()
            dtype = str(r.get('设备类型', '')).strip()
            if did and did not in ('', '[]', 'null'):
                if did not in id_to_info:
                    id_to_info[did] = {'设备类型': dtype, '设备指令号': did, '设备参数': str(r.get('设备参数','')).strip()}
    
    print(f'Unique devices: {len(id_to_info)}')
    
    # Generate descriptions
    devices = []
    for did in sorted(id_to_info.keys()):
        info = id_to_info[did]
        dtype = info['设备类型']
        desc = build_desc(did, dtype)
        info['设备指令功能说明'] = desc
        info['_text'] = f'{dtype} {desc} {did}'.strip()
        devices.append(info)
    
    # Save JSONL
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for d in devices:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f'Saved JSONL: {out_jsonl} ({len(devices)} entries)')
    
    # Build SQLite FTS
    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command_id TEXT UNIQUE NOT NULL,
            device_type TEXT,
            description TEXT,
            search_text TEXT
        )
    ''')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_cid ON devices(command_id)')
    
    # FTS5 table
    cur.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS devices_fts USING fts5(
            command_id, device_type, description, search_text,
            content=devices, content_rowid=id
        )
    ''')
    
    for d in devices:
        cur.execute(
            'INSERT OR REPLACE INTO devices (command_id, device_type, description, search_text) VALUES (?, ?, ?, ?)',
            (d['设备指令号'], d['设备类型'], d['设备指令功能说明'], d['_text'])
        )
    
    conn.commit()
    
    # Verify
    cur.execute('SELECT COUNT(*) FROM devices')
    count = cur.fetchone()[0]
    print(f'SQLite FTS: {count} records in {out_db}')
    
    conn.close()
    
    # Show sample descriptions
    print('\nSample descriptions:')
    for d in devices[:10]:
        print(f'  {d["设备指令号"]:45s} | {d["设备指令功能说明"]}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default=DEFAULT_CORPUS)
    p.add_argument('--output', default=DEFAULT_OUT)
    p.add_argument('--db', default=DEFAULT_DB)
    args = p.parse_args()
    
    if not os.path.exists(args.corpus):
        print(f'ERROR: {args.corpus} not found')
        sys.exit(1)
    
    build_index(args.corpus, args.output, args.db)
    print('\nDone!')
