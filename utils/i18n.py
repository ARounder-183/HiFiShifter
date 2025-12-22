import json
import os

class I18nManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(I18nManager, cls).__new__(cls)
            cls._instance.translations = {}
            cls._instance.current_lang = 'zh_CN'
            cls._instance.lang_map = {
                'zh_CN': '简体中文',
                'en_US': 'English'
            }
        return cls._instance

    def load_language(self, lang_code, assets_dir):
        self.current_lang = lang_code
        file_path = os.path.join(assets_dir, 'lang', f'{lang_code}.json')
        # Fallback to zh_CN if file doesn't exist, or just keep empty
        if not os.path.exists(file_path):
            print(f"Language file not found: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        except Exception as e:
            print(f"Error loading language file: {e}")
            self.translations = {}

    def get(self, key, default=None):
        return self.translations.get(key, default if default is not None else key)

i18n = I18nManager()
