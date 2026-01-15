#!/usr/bin/env python3
"""
修复 Gemini 账号的 project_id
通过 loadCodeAssist 或 onboardUser 获取真正的项目 ID
"""
import asyncio
import json
import logging
from account_manager import list_enabled_accounts, get_account, update_account
from gemini.auth import GeminiTokenManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gemini API 端点
GEMINI_API_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# Antigravity OAuth 凭据
GOOGLE_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"


async def fix_account_project_id(account: dict) -> bool:
    """修复单个账号的 project_id"""
    account_id = account['id']
    label = account.get('label', '未命名')
    refresh_token = account.get('refreshToken')

    if not refresh_token:
        logger.error(f"账号 {label} (ID: {account_id[:8]}...) 没有 refresh_token，跳过")
        return False

    logger.info(f"正在修复账号: {label} (ID: {account_id[:8]}...)")

    # 获取当前的 other 数据
    other = account.get('other', {})
    if isinstance(other, str):
        try:
            other = json.loads(other)
        except json.JSONDecodeError:
            other = {}
    if other is None:
        other = {}

    old_project_id = other.get('project', '无')
    logger.info(f"当前 project: {old_project_id}")

    # 创建 TokenManager 获取真正的 project_id
    token_manager = GeminiTokenManager(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        refresh_token=refresh_token,
        api_endpoint=GEMINI_API_ENDPOINT
    )

    try:
        # 获取真正的 project_id（会自动调用 onboardUser 如果需要）
        new_project_id = await token_manager.get_project_id()

        if new_project_id:
            logger.info(f"获取到新的 project_id: {new_project_id}")

            # 更新 other 字段
            other['project'] = new_project_id

            # 保存到数据库（同时更新 access_token）
            update_account(
                account_id,
                other=other,
                access_token=token_manager.access_token
            )
            logger.info(f"✅ 账号 {label} 的 project 已更新: {old_project_id} -> {new_project_id}")
            logger.info(f"✅ access_token 已同步更新")
            return True
        else:
            logger.error(f"❌ 无法获取账号 {label} 的 project_id")
            return False

    except Exception as e:
        logger.error(f"❌ 修复账号 {label} 时出错: {e}")
        return False


async def main():
    """主函数"""
    print("=" * 60)
    print("Gemini 账号 Project ID 修复工具")
    print("=" * 60)
    print()

    # 获取所有 Gemini 账号
    gemini_accounts = list_enabled_accounts(account_type='gemini')

    if not gemini_accounts:
        print("没有找到任何 Gemini 账号")
        return

    print(f"找到 {len(gemini_accounts)} 个 Gemini 账号:")
    for i, acc in enumerate(gemini_accounts):
        other = acc.get('other', {})
        if isinstance(other, str):
            try:
                other = json.loads(other)
            except:
                other = {}
        project_id = other.get('project', '无') if other else '无'
        print(f"  {i+1}. {acc.get('label', '未命名')} (ID: {acc['id'][:8]}...) - project: {project_id}")

    print()

    # 询问用户要修复哪个账号
    choice = input("请输入要修复的账号编号 (输入 'all' 修复所有，或输入编号如 '1'): ").strip()

    if choice.lower() == 'all':
        accounts_to_fix = gemini_accounts
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(gemini_accounts):
                accounts_to_fix = [gemini_accounts[idx]]
            else:
                print("无效的编号")
                return
        except ValueError:
            print("无效的输入")
            return

    print()
    print(f"将修复 {len(accounts_to_fix)} 个账号...")
    print()

    success_count = 0
    fail_count = 0

    for acc in accounts_to_fix:
        if await fix_account_project_id(acc):
            success_count += 1
        else:
            fail_count += 1
        print()

    print("=" * 60)
    print(f"修复完成: 成功 {success_count} 个，失败 {fail_count} 个")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
