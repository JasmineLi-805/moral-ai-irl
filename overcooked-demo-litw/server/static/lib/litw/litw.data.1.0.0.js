/*************************************************************
 * litw.data.1.0.0js
 *
 * Contains functions for writing LITW Study data operations
 * using the LITW REST API
 *
 * Dependencies: jQuery
 *
 * Author: LabintheWild DEV crew
 *
 * © Copyright 2021 LabintheWild
 * For questions about this file and permission to use
 * the code, contact us at info@labinthewild.org
 *************************************************************/

(function( exports ) {
    "use strict";

    var version = 1.0,
        params = {
            _isInitialized: false,
            participantId: 0,
            ipCountry: "not_fetched_or_initialized",
            ipCity: "not_fetched_or_initialized",
            userAgent: "not_fetched_or_initialized"
        },

        initialize = function() {
            let litw_locale = LITW.locale.getLocale() || "";

            if (!params._isInitialized) {
                params._isInitialized = true;
                params.participantId = uuidv4();
                params.userAgent = navigator.userAgent;
                let geoip_url = 'https://www.labinthewild.org/include/geoip.php';
                return $.getJSON(geoip_url, function(data) {
                    params.ipCity = data.city;
                    params.ipCountry = data.country;
                }).always(function() {
                    let data = {
                       contentLanguage: litw_locale,
                       city: params.ipCity,
                       country: params.ipCountry,
                       userAgent: params.userAgent
                    };
                    submitData(data,"litw:initialize");
                });
            }
        },

        uuidv4 = function() {
            return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(
            /[018]/g, c => (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16))
        },

        submitComments = function(data) {
            submitData(data,"study_client:comments")
        },
        submitDemographics = function(data) {
            submitData(data,"study_client:demographics")
        },
        submitStudyData = function(data) {
            submitData(data,"study_client:data")
        },
        submitData = function(data, dataType) {
            if (!params._isInitialized) {
                initialize();
            }
            data.user_id = getParticipantId();
            data.data_type = dataType;
            _submit(data, false);
        },
        getParticipantId = function() {
            return params.participantId;
        },
        getCountry = function() {
            return params.ipCountry;
        },
        getCity = function() {
            return params.ipCity;
        },
        isInitialized = function() {
            return params._isInitialized;
        },

        _submit = function(obj_data, finalAttempt) {
            //console.log(JSON.stringify(obj_data));
            $.ajax({
                url: 'data',
                type: 'PUT',
                data: JSON.stringify(obj_data),
                contentType: 'application/json'
            }).fail(function(e) {
                if (!finalAttempt) {
                    _submit(obj_data, true);
                }
            });
        }

    /**** PUBLIC METHODS ****/
    exports.data = {};
    exports.data.submitComments = submitComments;
    exports.data.submitDemographics = submitDemographics;
    exports.data.submitStudyData = submitStudyData;
    exports.data.submitData = submitData;
    exports.data.initialize = initialize;
    exports.data.getParticipantId = getParticipantId;
    exports.data.getCountry = getCountry;
    exports.data.getCity = getCity;
    exports.data.isInitialized = isInitialized;

})( window.LITW = window.LITW || {} );
