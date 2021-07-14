/*************************************************************
 * litw.tracking.0.1.js
 *
 * Contains functions for tracking participant progress over
 * the course of a study.
 * 
 * Dependencies: jQuery
 * 
 * © Copyright 2017 LabintheWild
 * For questions about this file and permission to use
 * the code, contact us at info@labinthewild.org
 *************************************************************/

(function( exports ) {
	"use strict";

	var version = 0.1,
	params = {
		DROPOUT_CODE_NORMAL_PROGRESSION: 0,
		DROPOUT_CODE_LEFT_STUDY: 1,
		currentDescription: ""
	},

	recordCheckpoint = function(description) {
		params.currentDescription = description;
		var data = {
			"description": params.currentDescription,
			"dropoutCode": params.DROPOUT_CODE_NORMAL_PROGRESSION
		};
		LITW.data.submitData(data, "tracking:checkpoint");
	},

    recordSharing = function(description) {
        var data = {
            "description": description
        };
        LITW.data.submitData(data, "tracking:sharing");
    },

    bindDropoutListener = function() {
        var data = {
            "dropoutCode": params.DROPOUT_CODE_LEFT_STUDY
        };
        window.onbeforeunload = function() {
            data.description = params.currentDescription;
            LITW.data.submitData(data, "tracking:dropout");
        };
	}


 	/**** PUBLIC METHODS ****/
 	exports.tracking = {};
	exports.tracking.recordCheckpoint = recordCheckpoint;
    exports.tracking.recordSharing = recordSharing;
	exports.tracking.bindDropoutListener = bindDropoutListener;

 })( window.LITW = window.LITW || {} );
