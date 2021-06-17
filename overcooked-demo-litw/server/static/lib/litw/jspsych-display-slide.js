/*************************************************************
 * jspsych-display-slide.js
 *
 * A jsPsych plugin that displays slides based on their name.
 *
 *
 * © Copyright 2020 LabintheWild
 * For questions about this file and permission to use
 * the code, contact us at info@labinthewild.org
 *************************************************************/

jsPsych.plugins["display-slide"] = (function() {

    var plugin = {};

    plugin.trial = function(display_element, trial) {
        $(".slide").hide().html('');
        var template_data = {};
        if(trial.template_data) {
            if(typeof(trial.template_data) === "function"){
                template_data = trial.template_data();
            } else {
                template_data = trial.template_data;
            }
        }
        display_element.html(trial.template(template_data));
        display_element.i18n();

        if(trial.setup) trial.setup();

        LITW.utils.showNextButton(function() {
            if(trial.finish) trial.finish();
            display_element.empty();
            jsPsych.finishTrial();
        }, {submitKeys: []});

        if(trial.show_next === false){
            $('#btn-next-page').hide();
        }

        //LITW.utils.showSlide(display_element[0].id);
        display_element.css('display', 'table');
        if(trial.name) {
            LITW.tracking.recordCheckpoint(trial.name);
        } else {
            LITW.tracking.recordCheckpoint(display_element[0].id);
        }
    };

    return plugin;

})();