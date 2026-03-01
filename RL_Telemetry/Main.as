Net::Socket@ tcpSocket;

void Main() {
    @tcpSocket = Net::Socket();
    tcpSocket.Connect("127.0.0.1", 9000);
    print("RL Telemetry (TCP): Skrypt zaladowany. Lecimy z UI Sequence.");
}

void Update(float dt) {
    auto app = cast<CTrackMania>(GetApp());
    if (app.CurrentPlayground is null || app.CurrentPlayground.GameTerminals.Length != 1) return;

    auto terminal = app.CurrentPlayground.GameTerminals[0];
    if (terminal.GUIPlayer is null) return;
    
    auto player = cast<CSmPlayer>(terminal.GUIPlayer);
    if (player is null) return;
    
    auto state = cast<CSmScriptPlayer>(player.ScriptAPI);
    if (state is null) return;

    bool is_finished = false;
    if (app.CurrentPlayground.UIConfigs.Length > 0) {
        auto ui_seq = app.CurrentPlayground.UIConfigs[0].UISequence;
        is_finished = (ui_seq == CGamePlaygroundUIConfig::EUISequence::Finish);
    }

    int cps_passed = state.RaceWaypointTimes.Length;
    float speed = state.Speed;
    int gear = state.EngineCurGear;
    
    vec3 pos = state.Position;
    vec3 vel = state.Velocity;

    string payload = "{\"speed\": " + speed + ", \"gear\": " + gear + 
                     ", \"pos_x\": " + pos.x + ", \"pos_y\": " + pos.y + ", \"pos_z\": " + pos.z + 
                     ", \"vel_x\": " + vel.x + ", \"vel_y\": " + vel.y + ", \"vel_z\": " + vel.z + 
                     ", \"cps_passed\": " + cps_passed + 
                     ", \"is_finished\": " + (is_finished ? "true" : "false") + "}\n";
    
    if (tcpSocket !is null) {
        tcpSocket.WriteRaw(payload);
    }
}