#!/usr/bin/env python
"""
添加设备记录的简单脚本
使用方法：python add_device.py
"""

import os
import sys
import django

# 设置Django环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartdoc.settings')

django.setup()

from backend.models import Device

def add_device():
    """添加一个测试设备"""
    try:
        # 检查是否已存在ID为1的设备
        existing_device = Device.objects.filter(id=1).first()
        if existing_device:
            print(f"设备已存在: ID={existing_device.id}, 名称={existing_device.name}, 类型={existing_device.type}")
            return existing_device
        
        # 创建新设备
        device = Device.objects.create(
            id=1,  # 明确指定ID为1
            name="测试摄像头",
            type="camera",
            model="Test Camera Model",
            status="online",
            config={"resolution": "1920x1080", "fps": 30}
        )
        
        print(f"成功创建设备: ID={device.id}, 名称={device.name}, 类型={device.type}")
        return device
        
    except Exception as e:
        print(f"创建设备失败: {e}")
        return None

def list_devices():
    """列出所有设备"""
    try:
        devices = Device.objects.all()
        print(f"\n当前数据库中的设备:")
        print("-" * 50)
        for device in devices:
            print(f"ID: {device.id}, 名称: {device.name}, 类型: {device.type}, 状态: {device.status}")
        print("-" * 50)
    except Exception as e:
        print(f"获取设备列表失败: {e}")

if __name__ == "__main__":
    print("开始添加设备...")
    
    # 添加设备
    device = add_device()
    
    # 列出所有设备
    list_devices()
    
    if device:
        print(f"\n✅ 设备添加成功！现在可以使用 device_id={device.id} 来创建告警了")
    else:
        print("\n❌ 设备添加失败") 