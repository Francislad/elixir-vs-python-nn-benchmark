import Config
config :nx, :default_backend, {EXLA.Backend, client: :cuda}
config :nx, :default_defn_options, [compiler: EXLA, client: :cuda]