#include "bgbot/match_equity.h"
#include "bgbot/cube.h"  // for CubeOwner
#include <algorithm>

namespace bgbot {

// ---------------------------------------------------------------------------
// Kazaross-XG2 Match Equity Table
// From: gnubg_src/met/Kazaross-XG2.xml
//
// MET_PRE[i][j] = MWC for player needing (i+1) points vs opponent needing (j+1) points.
// Row 0 = 1-away, Row 24 = 25-away. 0-indexed.
// ---------------------------------------------------------------------------

const std::array<std::array<float, MAX_MATCH_LENGTH>, MAX_MATCH_LENGTH> MET_PRE = {{
    // Row 0: 1-away
    {{ 0.50000f, 0.67736f, 0.75076f, 0.81436f, 0.84179f, 0.88731f, 0.90724f, 0.93250f, 0.94402f, 0.959275f, 0.966442f, 0.975534f, 0.979845f, 0.985273f, 0.987893f, 0.99114f, 0.99273f, 0.99467f, 0.99563f, 0.99679f, 0.99737f, 0.99807f, 0.99842f, 0.99884f, 0.99905f }},
    // Row 1: 2-away
    {{ 0.32264f, 0.50000f, 0.59947f, 0.66870f, 0.74359f, 0.79940f, 0.84225f, 0.87539f, 0.90197f, 0.923034f, 0.939311f, 0.952470f, 0.962495f, 0.970701f, 0.976887f, 0.98196f, 0.98580f, 0.98893f, 0.99129f, 0.99322f, 0.99466f, 0.99585f, 0.99675f, 0.99746f, 0.99802f }},
    // Row 2: 3-away
    {{ 0.24924f, 0.40053f, 0.50000f, 0.57150f, 0.64795f, 0.71123f, 0.76209f, 0.80468f, 0.84017f, 0.870638f, 0.894417f, 0.914831f, 0.930702f, 0.944426f, 0.954931f, 0.96399f, 0.97093f, 0.97687f, 0.98139f, 0.98522f, 0.98814f, 0.99062f, 0.99248f, 0.99407f, 0.99527f }},
    // Row 3: 4-away
    {{ 0.18564f, 0.33130f, 0.42850f, 0.50000f, 0.57732f, 0.64285f, 0.69924f, 0.74577f, 0.78799f, 0.824059f, 0.853955f, 0.879141f, 0.900233f, 0.918040f, 0.932657f, 0.94495f, 0.95499f, 0.96341f, 0.97021f, 0.97589f, 0.98044f, 0.98422f, 0.98726f, 0.98975f, 0.99174f }},
    // Row 4: 5-away
    {{ 0.15821f, 0.25641f, 0.35205f, 0.42268f, 0.50000f, 0.56635f, 0.62638f, 0.67786f, 0.72540f, 0.767055f, 0.802732f, 0.833654f, 0.859934f, 0.882866f, 0.902013f, 0.91847f, 0.93223f, 0.94397f, 0.95367f, 0.96189f, 0.96864f, 0.97432f, 0.97896f, 0.98283f, 0.98600f }},
    // Row 5: 6-away
    {{ 0.11269f, 0.20060f, 0.28877f, 0.35715f, 0.43365f, 0.50000f, 0.56261f, 0.61636f, 0.66787f, 0.713057f, 0.753427f, 0.788634f, 0.819569f, 0.846648f, 0.869999f, 0.89021f, 0.90756f, 0.92246f, 0.93508f, 0.94583f, 0.95488f, 0.96254f, 0.96894f, 0.97432f, 0.97879f }},
    // Row 6: 7-away
    {{ 0.09276f, 0.15775f, 0.23791f, 0.30076f, 0.37362f, 0.43739f, 0.50000f, 0.55480f, 0.60854f, 0.656283f, 0.700209f, 0.739054f, 0.774121f, 0.805203f, 0.832566f, 0.85659f, 0.87761f, 0.89591f, 0.91171f, 0.92535f, 0.93702f, 0.94703f, 0.95553f, 0.96276f, 0.96887f }},
    // Row 7: 8-away
    {{ 0.06750f, 0.12461f, 0.19532f, 0.25423f, 0.32214f, 0.38364f, 0.44520f, 0.50000f, 0.55442f, 0.603718f, 0.649899f, 0.691356f, 0.729447f, 0.763593f, 0.794397f, 0.82158f, 0.84578f, 0.86714f, 0.88589f, 0.90230f, 0.91658f, 0.92898f, 0.93968f, 0.94891f, 0.95682f }},
    // Row 8: 9-away
    {{ 0.05598f, 0.09803f, 0.15983f, 0.21201f, 0.27460f, 0.33213f, 0.39146f, 0.44558f, 0.50000f, 0.550196f, 0.597926f, 0.641481f, 0.682119f, 0.718927f, 0.752814f, 0.78301f, 0.81037f, 0.83483f, 0.85662f, 0.87591f, 0.89294f, 0.90791f, 0.92098f, 0.93240f, 0.94230f }},
    // Row 9: 10-away
    {{ 0.040725f, 0.076966f, 0.129362f, 0.175941f, 0.232945f, 0.286943f, 0.343717f, 0.396282f, 0.449804f, 0.500000f, 0.548547f, 0.593459f, 0.635880f, 0.674830f, 0.711113f, 0.74371f, 0.77375f, 0.80093f, 0.82543f, 0.84741f, 0.86703f, 0.88448f, 0.89991f, 0.91353f, 0.92550f }},
    // Row 10: 11-away
    {{ 0.033558f, 0.060689f, 0.105583f, 0.146045f, 0.197268f, 0.246573f, 0.299791f, 0.350101f, 0.402074f, 0.451453f, 0.500000f, 0.545552f, 0.589242f, 0.629736f, 0.667927f, 0.70303f, 0.73530f, 0.76494f, 0.79198f, 0.81648f, 0.83862f, 0.85849f, 0.87629f, 0.89214f, 0.90622f }},
    // Row 11: 12-away
    {{ 0.024466f, 0.047530f, 0.085169f, 0.120859f, 0.166346f, 0.211366f, 0.260946f, 0.308644f, 0.358519f, 0.406541f, 0.454448f, 0.500000f, 0.544068f, 0.585701f, 0.625259f, 0.66178f, 0.69610f, 0.72778f, 0.75703f, 0.78381f, 0.80826f, 0.83044f, 0.85051f, 0.86856f, 0.88476f }},
    // Row 12: 13-away
    {{ 0.020155f, 0.037505f, 0.069298f, 0.099767f, 0.140066f, 0.180431f, 0.225879f, 0.270553f, 0.317881f, 0.364120f, 0.410758f, 0.455932f, 0.500000f, 0.541943f, 0.582545f, 0.62036f, 0.65619f, 0.68966f, 0.72081f, 0.74963f, 0.77619f, 0.80054f, 0.82276f, 0.84295f, 0.86123f }},
    // Row 13: 14-away
    {{ 0.014727f, 0.029299f, 0.055574f, 0.081960f, 0.117134f, 0.153352f, 0.194797f, 0.236407f, 0.281073f, 0.325170f, 0.370264f, 0.414299f, 0.458057f, 0.500000f, 0.540750f, 0.57942f, 0.61634f, 0.65117f, 0.68391f, 0.71448f, 0.74290f, 0.76917f, 0.79339f, 0.81559f, 0.83586f }},
    // Row 14: 15-away
    {{ 0.012107f, 0.023113f, 0.045069f, 0.067343f, 0.097987f, 0.130001f, 0.167434f, 0.205603f, 0.247186f, 0.288887f, 0.332073f, 0.374741f, 0.417455f, 0.459250f, 0.500000f, 0.53916f, 0.57679f, 0.61261f, 0.64659f, 0.67859f, 0.70862f, 0.73664f, 0.76265f, 0.78669f, 0.80883f }},
    // Row 15: 16-away
    {{ 0.00886f, 0.01804f, 0.03601f, 0.05505f, 0.08153f, 0.10979f, 0.14341f, 0.17842f, 0.21699f, 0.25629f, 0.29697f, 0.33822f, 0.37964f, 0.42058f, 0.46084f, 0.50000f, 0.53796f, 0.57441f, 0.60929f, 0.64241f, 0.67376f, 0.70323f, 0.73084f, 0.75657f, 0.78046f }},
    // Row 16: 17-away
    {{ 0.00727f, 0.01420f, 0.02907f, 0.04501f, 0.06777f, 0.09244f, 0.12239f, 0.15422f, 0.18963f, 0.22625f, 0.26470f, 0.30390f, 0.34381f, 0.38366f, 0.42321f, 0.46204f, 0.50000f, 0.53676f, 0.57222f, 0.60618f, 0.63856f, 0.66925f, 0.69822f, 0.72542f, 0.75087f }},
    // Row 17: 18-away
    {{ 0.00533f, 0.01107f, 0.02313f, 0.03659f, 0.05603f, 0.07754f, 0.10409f, 0.13286f, 0.16517f, 0.19907f, 0.23506f, 0.27222f, 0.31034f, 0.34883f, 0.38739f, 0.42559f, 0.46324f, 0.50000f, 0.53574f, 0.57023f, 0.60336f, 0.63501f, 0.66510f, 0.69356f, 0.72038f }},
    // Row 18: 19-away
    {{ 0.00437f, 0.00871f, 0.01861f, 0.02979f, 0.04633f, 0.06492f, 0.08829f, 0.11411f, 0.14338f, 0.17457f, 0.20802f, 0.24297f, 0.27919f, 0.31609f, 0.35341f, 0.39071f, 0.42778f, 0.46426f, 0.50000f, 0.53475f, 0.56838f, 0.60073f, 0.63171f, 0.66122f, 0.68921f }},
    // Row 19: 20-away
    {{ 0.00321f, 0.00678f, 0.01478f, 0.02411f, 0.03811f, 0.05417f, 0.07465f, 0.09770f, 0.12409f, 0.15259f, 0.18352f, 0.21619f, 0.25037f, 0.28552f, 0.32141f, 0.35759f, 0.39382f, 0.42977f, 0.46525f, 0.50000f, 0.53387f, 0.56667f, 0.59830f, 0.62864f, 0.65760f }},
    // Row 20: 21-away
    {{ 0.00263f, 0.00534f, 0.01186f, 0.01956f, 0.03136f, 0.04512f, 0.06298f, 0.08342f, 0.10706f, 0.13297f, 0.16138f, 0.19174f, 0.22381f, 0.25710f, 0.29138f, 0.32624f, 0.36144f, 0.39664f, 0.43162f, 0.46613f, 0.50000f, 0.53303f, 0.56508f, 0.59603f, 0.62576f }},
    // Row 21: 22-away
    {{ 0.00193f, 0.00415f, 0.00938f, 0.01578f, 0.02568f, 0.03746f, 0.05297f, 0.07102f, 0.09209f, 0.11552f, 0.14151f, 0.16956f, 0.19946f, 0.23083f, 0.26336f, 0.29677f, 0.33075f, 0.36499f, 0.39927f, 0.43333f, 0.46697f, 0.50000f, 0.53226f, 0.56360f, 0.59391f }},
    // Row 22: 23-away
    {{ 0.00158f, 0.00325f, 0.00752f, 0.01274f, 0.02104f, 0.03106f, 0.04447f, 0.06032f, 0.07902f, 0.10009f, 0.12371f, 0.14949f, 0.17724f, 0.20661f, 0.23735f, 0.26916f, 0.30178f, 0.33490f, 0.36829f, 0.40170f, 0.43492f, 0.46774f, 0.50000f, 0.53153f, 0.56221f }},
    // Row 23: 24-away
    {{ 0.00116f, 0.00254f, 0.00593f, 0.01025f, 0.01717f, 0.02568f, 0.03724f, 0.05109f, 0.06760f, 0.08647f, 0.10786f, 0.13144f, 0.15705f, 0.18441f, 0.21331f, 0.24343f, 0.27458f, 0.30644f, 0.33878f, 0.37136f, 0.40397f, 0.43640f, 0.46847f, 0.50000f, 0.53086f }},
    // Row 24: 25-away
    {{ 0.00095f, 0.00198f, 0.00473f, 0.00826f, 0.01400f, 0.02121f, 0.03113f, 0.04318f, 0.05770f, 0.07450f, 0.09378f, 0.11524f, 0.13877f, 0.16414f, 0.19117f, 0.21954f, 0.24913f, 0.27962f, 0.31079f, 0.34240f, 0.37424f, 0.40609f, 0.43779f, 0.46914f, 0.50000f }},
}};

// Post-Crawford MET: trailer's MWC.
// Index 0 = trailer needs 1 point (DMP) = 0.5
// Index 1 = trailer needs 2 points = 0.48803
// etc.
const std::array<float, MAX_MATCH_LENGTH> MET_POST_CRAWFORD = {{
    0.500000f, 0.48803f, 0.32264f, 0.31002f, 0.19012f,
    0.18072f, 0.11559f, 0.10906f, 0.06953f, 0.065161f,
    0.042069f, 0.039060f, 0.025371f, 0.023428f, 0.015304f,
    0.014050f, 0.009240f, 0.008420f, 0.005560f, 0.005050f,
    0.003360f, 0.003030f, 0.002030f, 0.001820f, 0.001230f,
}};

// ---------------------------------------------------------------------------
// MET lookup functions
// ---------------------------------------------------------------------------

float get_met(int away1, int away2, bool is_post_crawford) {
    // Terminal conditions
    if (away1 <= 0) return 1.0f;  // Player already won
    if (away2 <= 0) return 0.0f;  // Opponent already won

    // Clamp to table size
    int a1 = std::min(away1, MAX_MATCH_LENGTH);
    int a2 = std::min(away2, MAX_MATCH_LENGTH);

    // Post-Crawford routing:
    // Only use the post-Crawford table when Crawford has already occurred
    // (is_post_crawford=true) AND someone is at 1-away.
    // The pre-Crawford table (MET_PRE) already has correct values for 1-away
    // positions — those values account for the upcoming Crawford game.
    if (is_post_crawford && (a1 == 1 || a2 == 1)) {
        if (a1 == 1) {
            // Player is 1-away (the leader). Opponent is the trailer.
            // MET_POST_CRAWFORD[a2-1] = trailer's MWC
            return 1.0f - MET_POST_CRAWFORD[a2 - 1];
        } else {
            // Opponent is 1-away (the leader). Player is the trailer.
            return MET_POST_CRAWFORD[a1 - 1];
        }
    }

    // Pre-Crawford lookup (0-indexed)
    return MET_PRE[a1 - 1][a2 - 1];
}

float get_met_after(int away1, int away2, int nPoints,
                    bool player_wins, bool is_crawford) {
    int new_away1 = player_wins ? away1 - nPoints : away1;
    int new_away2 = player_wins ? away2 : away2 - nPoints;
    // After a game ends, determine if the resulting state is post-Crawford.
    // If the current game IS the Crawford game, then after it ends we're
    // in the post-Crawford period (someone was already 1-away).
    // If the current game is NOT Crawford, and someone reaches 1-away from
    // this game's result, the Crawford game hasn't happened yet — use
    // pre-Crawford table (is_post_crawford=false).
    bool post_crawford = is_crawford;  // Crawford game just ended → post-Crawford
    return get_met(new_away1, new_away2, post_crawford);
}

float cubeless_mwc(const std::array<float, 5>& probs,
                   int away1, int away2, int cube_value, bool is_crawford) {
    // Decompose cumulative NN probs into exclusive probs:
    // P(win) = P(single_win) + P(gammon_win) + P(bg_win)
    // P(gammon_win) includes P(bg_win), so:
    float p_sw = probs[0] - probs[1];         // P(single win only)
    float p_gw = probs[1] - probs[2];         // P(gammon win, not bg)
    float p_bw = probs[2];                    // P(backgammon win)
    float p_sl = (1.0f - probs[0]) - probs[3]; // P(single loss only)
    float p_gl = probs[3] - probs[4];         // P(gammon loss, not bg)
    float p_bl = probs[4];                    // P(backgammon loss)

    // Weighted sum of 6 outcome MWCs
    float mwc = 0.0f;
    mwc += p_sw * get_met_after(away1, away2, cube_value, true, is_crawford);
    mwc += p_gw * get_met_after(away1, away2, 2 * cube_value, true, is_crawford);
    mwc += p_bw * get_met_after(away1, away2, 3 * cube_value, true, is_crawford);
    mwc += p_sl * get_met_after(away1, away2, cube_value, false, is_crawford);
    mwc += p_gl * get_met_after(away1, away2, 2 * cube_value, false, is_crawford);
    mwc += p_bl * get_met_after(away1, away2, 3 * cube_value, false, is_crawford);

    return mwc;
}

float eq2mwc(float equity, int away1, int away2, int cube_value,
             bool is_crawford) {
    float mwc_win = get_met_after(away1, away2, cube_value, true, is_crawford);
    float mwc_lose = get_met_after(away1, away2, cube_value, false, is_crawford);
    // Linear interpolation: equity=-1 → mwc_lose, equity=+1 → mwc_win
    return 0.5f * (equity * (mwc_win - mwc_lose) + (mwc_win + mwc_lose));
}

float mwc2eq(float mwc, int away1, int away2, int cube_value,
             bool is_crawford) {
    float mwc_win = get_met_after(away1, away2, cube_value, true, is_crawford);
    float mwc_lose = get_met_after(away1, away2, cube_value, false, is_crawford);
    float denom = mwc_win - mwc_lose;
    if (denom < 1e-10f) return 0.0f;  // Degenerate case
    return (2.0f * mwc - (mwc_win + mwc_lose)) / denom;
}

bool can_double_match(int away1, int away2, int cube_value,
                      CubeOwner owner, bool is_crawford) {
    // Crawford game: no doubling
    if (is_crawford) return false;

    // Post-Crawford detection
    bool post_crawford = (away1 == 1 || away2 == 1);

    // Dead cube: if the doubler can already win the match with a normal win
    // at the current cube value, doubling adds nothing.
    // More precisely: if away1 <= cube_value, the player wins the match with
    // any win at the current cube, so cube is dead from their perspective.
    // (They could still double strategically to threaten gammon, but standard
    // practice is that this is a dead cube.)
    // GNUbg: anScore[fMove] + nCube >= nMatchTo → cube dead
    if (away1 <= cube_value) return false;

    // Post-Crawford: the leader (1-away) cannot meaningfully double.
    // They already win the match with any win at cube=1.
    if (post_crawford && away1 == 1) return false;

    // Standard ownership check
    return (owner == CubeOwner::CENTERED || owner == CubeOwner::PLAYER);
}

float dp_mwc(int away1, int away2, int cube_value, bool is_crawford) {
    // When the opponent passes, the player wins cube_value points.
    return get_met_after(away1, away2, cube_value, true, is_crawford);
}

} // namespace bgbot
