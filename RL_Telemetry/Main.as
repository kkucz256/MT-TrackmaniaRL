Net::Socket@ tcpSocket;
uint lastConnectAttempt = 0;

void Main() {
    print("RL Telemetry: Skrypt zaladowany. Czekam na proces Trenera...");
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

    if (tcpSocket is null) {
        if (Time::Now - lastConnectAttempt > 2000) {
            @tcpSocket = Net::Socket();
            tcpSocket.Connect("127.0.0.1", 9000);
            lastConnectAttempt = Time::Now;
        }
    }

    vec3 forward_vec = state.AimDirection;
    vec3 right_vec = vec3(forward_vec.z, 0.0, -forward_vec.x);
    
    float len = Math::Sqrt(right_vec.x * right_vec.x + right_vec.z * right_vec.z);
    if (len > 0.001) {
        right_vec.x /= len;
        right_vec.z /= len;
    } else {
        right_vec = vec3(1.0, 0.0, 0.0);
    }
    
    vec3 vel = state.Velocity;
    float slip_forward = Math::Dot(vel, forward_vec);
    float slip_side = Math::Dot(vel, right_vec);
    
    int cps_passed = state.RaceWaypointTimes.Length;
    float speed = state.Speed;
    int gear = state.EngineCurGear;
    bool is_finished = false;
    if (app.CurrentPlayground.UIConfigs.Length > 0) {
        auto ui_seq = app.CurrentPlayground.UIConfigs[0].UISequence;
        is_finished = (ui_seq == CGamePlaygroundUIConfig::EUISequence::Finish);
    }
    
    float rpm = state.EngineRpm; 
    string is_finished_str = is_finished ? "true" : "false";

    string payload = "{\"speed\": " + speed + ", \"gear\": " + gear + 
                     ", \"rpm\": " + rpm + 
                     ", \"slip_forward\": " + slip_forward + ", \"slip_side\": " + slip_side + 
                     ", \"cps_passed\": " + cps_passed +
                     ", \"is_finished\": " + is_finished_str + "}\n";
    
    if (tcpSocket !is null) {
        tcpSocket.WriteRaw(payload);
    }
}