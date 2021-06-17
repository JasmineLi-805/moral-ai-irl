// Persistent network connection that will be used to transmit real-time data
let socket = io();

// MAKE SURE that that "playerOne" is always the BOT!
let games_config = {
    "tutorial": {
        "game_name" : "litw_tutorial",
        "layouts" : ["tutorial_0"],
        "playerZero" : "human",
        "playerOne" : "TutorialAI",
        "mdp_params" : {
            "delivery_reward" : 10,
        }
    },
    "tutorial_coop": {
        "game_name" : "litw_tutorial_coop",
        "layouts" : ["tutorial_0_coop"],
        "playerZero" : "human",
        "playerOne" : "TutorialAI",
        "mdp_params" : {
            "delivery_reward" : 10,
        }
    },
    "mai_left": {
        "game_name" : "litw_cook",
        "layouts" : ["mai_separate_coop_left"],
        "playerZero" : "human",
        "playerOne" : "right_coop",
        "gameTime" : 60,
        "mdp_params" : {
            "delivery_reward" : 10,
        }
    },
    "mai_right": {
        "game_name" : "litw_cook",
        "layouts" : ["mai_separate_coop_right"],
        "playerZero" : "human",
        "playerOne" : "left_coop",
        "gameTime" : 60,
        "mdp_params" : {
            "delivery_reward" : 10,
        }
    },
};
let study_timeline = [];
let templates = {
    consent: {
        resource: 'static/templates/litw-consent.html',
        template: null
    },
    demographics: {
        resource: 'static/templates/litw-demographics.html',
        template: null
    },
    tutorial1_1: {
        resource: 'static/templates/litw-tutorial-step1_1.html',
        template: null
    },
    tutorial1_2: {
        resource: 'static/templates/litw-tutorial-step1_2.html',
        template: null
    },
    tutorial2: {
        resource: 'static/templates/litw-tutorial-step2.html',
        template: null
    },
    tutorial3: {
        resource: 'static/templates/litw-tutorial-step3.html',
        template: null
    },
    rounds_inst: {
        resource: 'static/templates/litw-round-instructions.html',
        template: null
    },
    rounds: {
        resource: 'static/templates/litw-round.html',
        template: null
    },
    comments: {
        resource: 'static/templates/litw-comments.html',
        template: null
    },
    results: {
        resource: 'static/templates/litw-results.html',
        template: null
    },
    results_footer: {
        resource: 'static/templates/litw-results-footer.html',
        template: null
    }
};
let study_data = {};


/* Socket event handlers */
socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#overcooked').append(`<h4>Sorry, study creation failed with error: ${JSON.stringify(err)}</>`);
    $('#try-again').show();
    $('#try-again').attr("disabled", false);
});

socket.on("connect", function() {
    console.log("SOCKET CONNECTED!");
});

socket.on('start_game', function(data) {
    console.log(`STARTING GAME: ${JSON.stringify(data)}`);
    graphics_config = {
        container_id : "overcooked",
        start_info : data.start_info
    };
    enable_key_listener();
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
    let game_data = data.data;
    graphics_end();
    disable_key_listener();
    setLastGameData(game_data);
    $('#btn-next-page').click();
});

socket.on('state_pong', function(data) {
    let cur_state = data['state'];
    delete cur_state['all_orders'];
    delete cur_state['bonus_orders'];
    drawState(cur_state);
});

socket.on('end_game', function (data){
    end_game(data);
});

function start_game(game_name) {
    let game_config = get_game_config(game_name);
    console.log("STARTING GAME: " + JSON.stringify(game_config));
    socket.emit("join", game_config);
}

function get_game_config(config_name) {
    let game_config = null;
    if(config_name in games_config){
        let stored_config = JSON.parse(JSON.stringify(games_config[config_name]));
        let game_name = stored_config['game_name'];
        delete stored_config['game_name'];
        stored_config.litw_uuid = LITW.data.getParticipantId();
        game_config = {
            "params": stored_config,
            "game_name": game_name
        }
    }
    return game_config;
}

function end_game(data) {
    // Hide game data and display game-over html
    graphics_end();
    disable_key_listener();
    console.log("END_STUDY: " + JSON.stringify(data));
}

function setLastGameData(game_data) {
    let size = study_data.games.length;
    if(size>0) {
        study_data.games[size-1].data = game_data;
    }
}

function getLastGameData() {
    let size = study_data.games.length;
    if(size>0) {
        return study_data.games[size - 1].data;
    } else {
        return null;
    }
}

/* Game Key Event Listener */

function enable_key_listener() {
    $(document).on('keydown', function(e) {
        let action = 'STAY'
        switch (e.which) {
            case 37: // left
                action = 'LEFT';
                break;

            case 38: // up
                action = 'UP';
                break;

            case 39: // right
                action = 'RIGHT';
                break;

            case 40: // down
                action = 'DOWN';
                break;

            case 32: //space
                action = 'SPACE';
                break;

            default: // exit this handler for other keys
                return;
        }
        e.preventDefault();
        socket.emit('action', { 'action' : action });
    });
};

function disable_key_listener() {
    $(document).off('keydown');
};


/* LITW STUDY CONFIGURATION */

function _init_litw(){
    LITW.data.initialize().then( function () {
        study_data.litw_uuid = LITW.data.getParticipantId();
        study_data.country = LITW.data.getCountry();
        study_data.city = LITW.data.getCity();
    });
    study_data.locale = LITW.locale.getLocale();
    study_data.games = [];
}

function start_study(){
    _init_litw();

    const async_load = async (template_names) => {
        const promises = template_names.map(load_template);
        await Promise.all(promises);
    };

    $.i18n().locale = study_data.locale;
    $.i18n().load({
        'en': 'static/templates/i18n/en.json',
    }).done(function(){
        $('head').i18n();
        console.log($.i18n('litw-study-title'));
        $('body').i18n();
        let template_names = Object.keys(templates);
        async_load(template_names).then( function(){
            configure_study();
            jsPsych.init({
                timeline: study_timeline
            });
        });
    });
}

function load_template(template_name) {
    Handlebars.registerHelper('times', function(n, block) {
        let accum = '';
        for(let i = 1; i <= n; ++i)
            accum += block.fn(i);
        return accum;
    });

    return $.get(templates[template_name].resource, function(html){
        templates[template_name].template = Handlebars.compile(html);
        // console.log(`LOADED ${JSON.stringify(templates[template_name].template)}`);
    })
}

function configure_study() {
    study_timeline.push({
        name: "informed_consent",
        type: "display-slide",
        template: templates.consent.template,
        display_element: $("#informed_consent"),
        show_next: false
    });
    study_timeline.push({
        name: "demographics",
        type: "display-slide",
        template: templates.demographics.template,
        display_element: $("#demographics"),
        show_next: false,
        finish: function(){
            let form_data = $('#demographicsForm').alpaca().getValue();
            form_data['time_elapsed'] = getSlideTime();
            LITW.data.submitDemographics(form_data);
        }
    });
    study_timeline.push({
        name: "tutorial1_1",
        type: "display-slide",
        template: templates.tutorial1_1.template,
        display_element: $("#tutorial"),
        show_next: true
    });
    study_timeline.push({
        name: "tutorial1_2",
        type: "display-slide",
        template: templates.tutorial1_2.template,
        display_element: $("#tutorial"),
        show_next: true
    });
    study_timeline.push({
        name: "tutorial2",
        type: "display-slide",
        template: templates.tutorial2.template,
        display_element: $("#tutorial"),
        show_next: false,
        setup: function (){
            start_game('tutorial');
        }
    });
    study_timeline.push({
        name: "tutorial3",
        type: "display-slide",
        template: templates.tutorial3.template,
        display_element: $("#tutorial"),
        show_next: false,
        setup: function (){
            start_game('tutorial_coop');
        }
    });
    study_timeline.push({
        name: "round1-instructions",
        type: "display-slide",
        display_element: $("#round-instructions"),
        show_next: true,
        template: templates.rounds_inst.template,
        template_data: {
            'header': $.i18n('litw-round-1-inst-header'),
            'instructions':[
                $.i18n('litw-round-1-inst-p1'),
                $.i18n('litw-round-1-inst-p2'),
                $.i18n('litw-round-1-inst-p3'),
                $.i18n('litw-round-1-inst-p4')
            ],
            'image': '/static/images/rounds/round-layout1.png'
        }
    });
    study_timeline.push({
        name: "round1",
        type: "display-slide",
        display_element: $("#round-game"),
        show_next: false,
        template: templates.rounds.template,
        template_data: {
            'header': $.i18n('litw-round-1-header')
        },
        setup: function (){
            study_data.games.push({name: 'round1'});
            start_game('mai_left');
        },
        finish: function (){
            LITW.data.submitStudyData(getLastGameData());
        }
    });

    study_timeline.push({
        name: "round2-instructions",
        type: "display-slide",
        display_element: $("#round-instructions"),
        show_next: true,
        template: templates.rounds_inst.template,
        template_data: {
            'header': $.i18n('litw-round-2-inst-header'),
            'instructions':[
                $.i18n('litw-round-2-inst-p1'),
                $.i18n('litw-round-2-inst-p2-npriv'),
                $.i18n('litw-round-2-inst-p3'),
                $.i18n('litw-round-2-inst-p4')
            ],
            'image': '/static/images/rounds/round-layout1.png'
        }
    });

    study_timeline.push({
        name: "round2",
        type: "display-slide",
        display_element: $("#round-game"),
        show_next: false,
        template: templates.rounds.template,
        template_data: {
            'header': $.i18n('litw-round-2-header')
        },
        setup: function (){
            study_data.games.push({name: 'round2'});
            start_game('mai_right');
        },
        finish: function (){
            LITW.data.submitStudyData(getLastGameData());
        }
    });

    study_timeline.push({
        name: "round3-instructions",
        type: "display-slide",
        display_element: $("#round-instructions"),
        show_next: true,
        template: templates.rounds_inst.template,
        template_data: {
            'header': $.i18n('litw-round-3-inst-header'),
            'instructions':[
                $.i18n('litw-round-3-inst-p1'),
                $.i18n('litw-round-3-inst-p2'),
                $.i18n('litw-round-3-inst-p3'),
                $.i18n('litw-round-3-inst-p4')
            ],
            'image': '/static/images/rounds/round-layout1.png'
        }
    });
    study_timeline.push({
        name: "round3",
        type: "display-slide",
        display_element: $("#round-game"),
        show_next: false,
        template: templates.rounds.template,
        template_data: {
            'header': $.i18n('litw-round-3-header')
        },
        setup: function (){
            study_data.games.push({name: 'round3'});
            start_game('mai_left');
        },
        finish: function (){
            LITW.data.submitStudyData(getLastGameData());
        }
    });

    study_timeline.push({
        name: "comments",
        type: "display-slide",
        display_element: $("#comments"),
        show_next: true,
        template: templates.comments.template,
        finish: function(){
            let comments = $('#commentsForm').alpaca().getValue();
            if (Object.keys(comments).length > 0) {
                comments['time_elapsed'] = getSlideTime();
                LITW.data.submitComments(comments);
            }
        }
    });

    study_timeline.push({
        name: "download-data",
        type: "call-function",
        func: function(){
            let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(study_data));
            let downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "study_data.json");
            document.body.appendChild(downloadAnchorNode); // required for firefox
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        }
    });

    study_timeline.push({
        name: "results",
        type: "call-function",
        func: showResults
    });
}

function getSlideTime() {
    let data_size = jsPsych.data.getData().length;
    if( data_size > 0 ) {
        return jsPsych.totalTime() - jsPsych.data.getLastTrialData().time_elapsed;
    } else {
        return jsPsych.totalTime();
    }
}

function showResults(){
    if (study_data.games.length < 3){
        //FAKE DATA FOR TESTING
        let fake_data = {
            score: [20,10],
            agent_coop_count: {
                received: 5,
                provided: 5
            }
        }
        for(let size=study_data.games.length; size<3; size++) {
            study_data.games.push({data: JSON.parse(JSON.stringify(fake_data))});
        }
    }
    let template_data = {
        message: $.i18n('litw-result-thanks')
    }
    console.log(study_data)
    let p_score = 0;
    let o_score = 0;
    let p_coop = 0;
    let o_coop = 0;
    for(let round=1; round <= 3; round++){
        let round_data = study_data.games[round-1].data
        p_score += round_data.score[0];
        o_score += round_data.score[1];
        p_coop += round_data.agent_coop_count.received;
        o_coop += round_data.agent_coop_count.provided;
    }
    template_data.score_p_avg = Math.floor(p_score/3);
    template_data.score_o_avg = Math.floor(o_score/3);
    template_data.coop_p_avg = Math.floor(p_coop/2);
    template_data.coop_o_avg = Math.floor(o_coop);
    let supercook = template_data.score_p_avg > template_data.score_o_avg;
    let supercoop = template_data.coop_p_avg > template_data.coop_o_avg;
    if(supercook) template_data.message = $.i18n('litw-result-supercook');
    if(supercoop) template_data.message = $.i18n('litw-result-highfive');
    if(supercook && supercoop) template_data.message = $.i18n('litw-result-both');
    $("#results").html(templates.results.template({
        results: template_data
    }));

    $("#results-footer").html(templates.results_footer.template({
        //TODO fix this before launching!
        share_url: "https://cook.moralai.org/litw",
        share_title: $.i18n('litw-irb-header'),
        share_text: $.i18n('litw-template-title'),
        more_litw_studies: [{
            study_url: "https://reading.labinthewild.org/",
            study_logo: "http://labinthewild.org/images/reading-assessment.jpg",
            study_slogan: $.i18n('litw-results-more-study1-slogan'),
            study_description: $.i18n('litw-results-more-study1-description'),
        },
        {
            study_url: "https://litw-sci-scomm.azurewebsites.net/LITW/consent",
            study_logo: "http://labinthewild.org/images/sci-comm-img.png",
            study_slogan: $.i18n('litw-results-more-study2-slogan'),
            study_description: $.i18n('litw-results-more-study2-description'),
        }]
    }));
    $("#results").i18n();
    LITW.utils.showSlide("results");
}

$(function() {
    start_study();
})