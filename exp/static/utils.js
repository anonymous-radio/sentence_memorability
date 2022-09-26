// constants
// change the following prior to real experiment: trials_per_block, debug_mode, completion_code
const use_flask = false;
const trials_per_block = 85; // 85;
const stimulus_duration = 2000;
const fixation_duration = 1400;
const completion_code = "XXXXXX";
const n_back_base = 20;
const vigilance_repeat_back_range = [1, 7];
const vigilance_frequency = 0.25;
const repeat_list_shuffle_block_size = 2;
const breaks_per_exp = 12;
const break_max_len = 180;
const num_lists = 12;
const debug_mode = false;

// instructions
const task_instructions = [
    "<p>TASK INSTRUCTIONS.</p> <p>This experiment should take around 60 minutes, with a quick break approximately every 5 minutes. " +
    "The progress bar at the top of the screen will indicate your progress in the experiment. " +
    "Do NOT reload the page during the experiment, as this will cause you to lose your progress.</p>",

    "<p>TASK INSTRUCTIONS.</p> <p>It is important that you pay attention during this study. " +
    "Please note that there are some trials where we expect everyone to be able to answer correctly. " +
    "<b>If you don't answer most of these correctly, you will not get paid.</b></p>",

    "<p>TASK INSTRUCTIONS.</p> <p>Once you complete the experiment, you will be shown your completion code. " +
    "Be careful not to accidentally exit the experiment before getting your completion code.</p>",

    "<p>TASK INSTRUCTIONS.</p> <p>You will see a series of sentences, one on each screen. " +
    "Press SPACE if you have seen the sentence before at <b>ANY</b> point during the study. " +
    "You may press SPACE while the sentence is on the screen or during the waiting period after (when the + sign is on the screen).</p>",

    "<p>TASK INSTRUCTIONS.</p> <p>Pressing SPACE will make a chime noise. " +
    "This is just to let you know that you've pressed SPACE, but the sound is unrelated to the correctness of your response.</p>",

    "<p>TASK INSTRUCTIONS.</p> <p>You will now see a few practice examples. Remember: press SPACE if you see any repeated sentence. </p>",
]

// formatting function
function format(s) {
    return '<span style="font-size:40px;">' + s + '</span>';
}

// saving data
function saveData(name, data) {
    var xhr = new XMLHttpRequest();
    if (use_flask) {
        xhr.open('POST', "{{url_for('save')}}");
    } else {
        xhr.open('POST', "../static/write_data.php");
    }
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({ filename: name, filedata: data }));
}
