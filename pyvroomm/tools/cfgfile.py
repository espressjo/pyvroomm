class ConfigParser:
    """
    A configuration file parser that supports comments and flexible whitespace.
    Supports extraction of int, float, string, and boolean values with type validation.
    """
    
    def __init__(self, config_file_path=None):
        """
        Initialize the parser with optional config file path.
        
        Args:
            config_file_path (str, optional): Path to the configuration file
        """
        self.config_data = {}
        if config_file_path:
            self.load_config(config_file_path)
    
    def load_config(self, file_path):
        """
        Load and parse configuration file.
        
        Args:
            file_path (str): Path to the configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file has invalid format
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self._parse_config(file.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    def load_config_from_string(self, config_string):
        """
        Load configuration from a string.
        
        Args:
            config_string (str): Configuration content as string
        """
        self._parse_config(config_string)
    
    def _parse_config(self, content):
        """
        Parse configuration content and populate config_data dictionary.
        
        Args:
            content (str): Raw configuration file content
        """
        self.config_data = {}
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Remove comments (everything after #)
            comment_pos = line.find('#')
            if comment_pos != -1:
                line = line[:comment_pos]
            
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Find the separator (=, :, or whitespace)
            separator_pos = -1
            separator = None
            
            # Check for = or : separators first
            for sep in ['=', ':']:
                pos = line.find(sep)
                if pos != -1:
                    separator_pos = pos
                    separator = sep
                    break
            
            # If no = or : found, look for whitespace separator
            if separator_pos == -1:
                for i, char in enumerate(line):
                    if char.isspace():
                        separator_pos = i
                        separator = 'whitespace'
                        break
            
            if separator_pos == -1:
                raise ValueError(f"Invalid configuration format at line {line_num}: '{line.strip()}'")
            
            # Extract key and value
            key = line[:separator_pos].strip()
            if separator == 'whitespace':
                # For whitespace separator, we need to find where the value starts
                value_start = separator_pos
                while value_start < len(line) and line[value_start].isspace():
                    value_start += 1
                value = line[value_start:].strip() if value_start < len(line) else ''
            else:
                value = line[separator_pos + 1:].strip()
            
            if not key:
                raise ValueError(f"Empty key at line {line_num}")
            
            self.config_data[key] = value
    
    def _get_raw_value(self, key):
        """
        Get raw string value for a key.
        
        Args:
            key (str): Configuration key
            
        Returns:
            str: Raw string value
            
        Raises:
            KeyError: If key doesn't exist
        """
        if key not in self.config_data:
            raise KeyError(f"Key '{key}' not found in configuration")
        return self.config_data[key]
    
    def get_string(self, key, default=None):
        """
        Get string value for a key.
        
        Args:
            key (str): Configuration key
            default (str, optional): Default value if key not found
            
        Returns:
            str: String value
            
        Raises:
            KeyError: If key doesn't exist and no default provided
        """
        try:
            value = self._get_raw_value(key)
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            return value
        except KeyError:
            if default is not None:
                return default
            raise
    
    def get_int(self, key, default=None):
        """
        Get integer value for a key with validation.
        
        Args:
            key (str): Configuration key
            default (int, optional): Default value if key not found
            
        Returns:
            int: Integer value
            
        Raises:
            KeyError: If key doesn't exist and no default provided
            ValueError: If value cannot be converted to int
        """
        try:
            raw_value = self._get_raw_value(key)
            try:
                return int(raw_value)
            except ValueError:
                raise ValueError(f"Value '{raw_value}' for key '{key}' is not a valid integer")
        except KeyError:
            if default is not None:
                return default
            raise
    
    def get_float(self, key, default=None):
        """
        Get float value for a key with validation.
        
        Args:
            key (str): Configuration key
            default (float, optional): Default value if key not found
            
        Returns:
            float: Float value
            
        Raises:
            KeyError: If key doesn't exist and no default provided
            ValueError: If value cannot be converted to float
        """
        try:
            raw_value = self._get_raw_value(key)
            try:
                return float(raw_value)
            except ValueError:
                raise ValueError(f"Value '{raw_value}' for key '{key}' is not a valid float")
        except KeyError:
            if default is not None:
                return default
            raise
    
    def get_bool(self, key, default=None):
        """
        Get boolean value for a key with validation.
        Accepts: true/false, yes/no, 1/0, on/off (case insensitive)
        
        Args:
            key (str): Configuration key
            default (bool, optional): Default value if key not found
            
        Returns:
            bool: Boolean value
            
        Raises:
            KeyError: If key doesn't exist and no default provided
            ValueError: If value cannot be converted to bool
        """
        try:
            raw_value = self._get_raw_value(key).lower().strip()
            
            true_values = {'true', 'yes', '1', 'on', 'enable', 'enabled'}
            false_values = {'false', 'no', '0', 'off', 'disable', 'disabled'}
            
            if raw_value in true_values:
                return True
            elif raw_value in false_values:
                return False
            else:
                raise ValueError(f"Value '{raw_value}' for key '{key}' is not a valid boolean. "
                               f"Use: {', '.join(true_values | false_values)}")
        except KeyError:
            if default is not None:
                return default
            raise
    
    def has_key(self, key):
        """
        Check if a key exists in the configuration.
        
        Args:
            key (str): Configuration key
            
        Returns:
            bool: True if key exists, False otherwise
        """
        return key in self.config_data
    
    def get_all_keys(self):
        """
        Get all configuration keys.
        
        Returns:
            list: List of all keys
        """
        return list(self.config_data.keys())
    
    def __str__(self):
        """String representation of the configuration."""
        return f"ConfigParser with {len(self.config_data)} keys: {list(self.config_data.keys())}"


# Example usage and testing
if __name__ == "__main__":
    # Example configuration content
    sample_config = """
    # Database configuration
    db_host = localhost
    db_port:    5432    # PostgreSQL default port
    db_name   =   "myapp_db"
    db_ssl_enabled = true
    
    # Application settings
    debug_mode: yes
    max_connections = 100
    timeout_seconds   30.5
    app_name = "My Application"
    
    # Feature flags
    feature_x_enabled = false
    feature_y_enabled    =    1    # Enabled
    log_level = info    # Can be: debug, info, warn, error
    """
    
    # Create parser and load configuration
    parser = ConfigParser()
    parser.load_config_from_string(sample_config)
    
    # Test different data types
    print("=== Configuration Parser Demo ===")
    print(f"Database host: {parser.get_string('db_host')}")
    print(f"Database port: {parser.get_int('db_port')}")
    print(f"Database name: {parser.get_string('db_name')}")
    print(f"SSL enabled: {parser.get_bool('db_ssl_enabled')}")
    print(f"Debug mode: {parser.get_bool('debug_mode')}")
    print(f"Max connections: {parser.get_int('max_connections')}")
    print(f"Timeout: {parser.get_float('timeout_seconds')}")
    print(f"App name: {parser.get_string('app_name')}")
    print(f"Feature X enabled: {parser.get_bool('feature_x_enabled')}")
    print(f"Feature Y enabled: {parser.get_bool('feature_y_enabled')}")
    
    # Test default values
    print(f"Non-existent key with default: {parser.get_string('missing_key', 'default_value')}")
    
    # Show all keys
    print(f"All keys: {parser.get_all_keys()}")