{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.python311Packages.schedule
    pkgs.python311Packages.pytz
    pkgs.python311Packages.python-dotenv
    pkgs.python311Packages.pandas
    pkgs.python311Packages.yfinance
    pkgs.python311Packages.requests
  ];
}


