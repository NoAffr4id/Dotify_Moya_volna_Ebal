use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device { Mobile, Desktop, Web }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activity { Idle, Commute, Focus, Party, Sleep }

impl Activity {
    /// Множитель для «энергичных» треков (bpm, energy high).
    pub fn energy_bias(self) -> f32 {
        match self {
            Activity::Idle    => 1.00,
            Activity::Commute => 1.10,
            Activity::Focus   => 0.85,
            Activity::Party   => 1.30,
            Activity::Sleep   => 0.50,
        }
    }

    /// Множитель для вокальных треков.
    pub fn vocal_bias(self) -> f32 {
        match self {
            Activity::Idle    => 1.00,
            Activity::Commute => 1.10,
            Activity::Focus   => 0.70,
            Activity::Party   => 1.20,
            Activity::Sleep   => 0.80,
        }
    }

    /// Множитель для новизны (explore).
    pub fn novelty_bias(self) -> f32 {
        match self {
            Activity::Idle    => 1.00,
            Activity::Commute => 0.90,
            Activity::Focus   => 0.70,
            Activity::Party   => 1.10,
            Activity::Sleep   => 0.60,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TemporalContext {
    pub hour:    u8,
    pub weekday: u8,
    pub device:  Device,
    pub activity: Activity,
    pub focused: bool,
}

impl Default for TemporalContext {
    fn default() -> Self {
        Self { hour: 0, weekday: 0, device: Device::Desktop, activity: Activity::Idle, focused: true }
    }
}

impl TemporalContext {
    pub fn bucket(&self) -> (u8, Device) { (self.hour, self.device) }
    pub fn is_new_session(&self, since_last_event_sec: f32, break_hours: f32) -> bool {
        since_last_event_sec > break_hours * 3600.0
    }
}