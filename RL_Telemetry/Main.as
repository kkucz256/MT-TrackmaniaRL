Net::Socket@ tcpSocket;
uint lastConnectAttempt = 0;

void Main() {
    print("RL Telemetry: Script loaded. Waiting for RL Trainer...");
}

void Update(float dt) {
    auto app = cast<CTrackMania>(GetApp());
    
    CSmScriptPlayer@ state = null;
    bool is_finished = false;

    if (app.CurrentPlayground !is null && app.CurrentPlayground.GameTerminals.Length == 1) {
        auto terminal = app.CurrentPlayground.GameTerminals[0];
        if (terminal.GUIPlayer !is null) {
            auto player = cast<CSmPlayer>(terminal.GUIPlayer);
            if (player !is null) {
                @state = cast<CSmScriptPlayer>(player.ScriptAPI);
            }
        }
        
        if (app.CurrentPlayground.UIConfigs.Length > 0) {
            auto ui_seq = app.CurrentPlayground.UIConfigs[0].UISequence;
            is_finished = (ui_seq == CGamePlaygroundUIConfig::EUISequence::Finish);
        }
    }

    if (state is null) {
        if (tcpSocket !is null) {
            tcpSocket.WriteRaw("{\"is_loading\": true}\n");
        }
        return;
    }

    if (tcpSocket is null) {
        if (Time::Now - lastConnectAttempt > 2000) {
            @tcpSocket = Net::Socket();
            tcpSocket.Connect("127.0.0.1", 9000);
            lastConnectAttempt = Time::Now;
        }
    }

    vec3 forward_vec = state.AimDirection;
    vec3 vel = state.Velocity;
    float slip_forward = Math::Dot(vel, forward_vec);
    
    int cps_passed = state.RaceWaypointTimes.Length;
    float speed = state.Speed;
    int gear = state.EngineCurGear;
    float rpm = state.EngineRpm; 
    
    string is_finished_str = is_finished ? "true" : "false";

    string payload = "{\"speed\": " + speed + ", \"gear\": " + gear + 
                     ", \"rpm\": " + rpm + 
                     ", \"slip_forward\": " + slip_forward + 
                     ", \"cps_passed\": " + cps_passed +
                     ", \"is_finished\": " + is_finished_str + 
                     ", \"is_loading\": false}\n";
    
    if (tcpSocket !is null) {
        tcpSocket.WriteRaw(payload);
    }
}