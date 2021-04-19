document.addEventListener("DOMContentLoaded", function(){
	let increments = [0, 0, 0]

	let question_input = $('#question-input');
	let context_input = $('#context-input');
	let question_type_input = $('#question-type-input');
	let predicted_answer_input = $('#answer-input');
	let expected_answer_input = $('#expected-answer-input');


	const yesno_examples = [
		["Can Freund's complete adjuvant induce arthritis?", "Rheumatoid arthritis (RA) was induced by Freunds Complete Adjuvant (FCA; 1 mg/0.1 ml paraffin oil), injected subcutaneously on days 0, 30 and 40", ["Yes"]],
		["Should Lubeluzole be used for treatment of ischemic stroke?", "Lubeluzole showed promising neuroprotective effects in animal stroke models, but failed to show benefits in acute ischemic stroke in humans.", ["No"]],
		["Is PTEN a tumour suppressor?", "Genomic aberrations of the PTEN tumour suppressor gene are among the most common in prostate cancer.", ["Yes"]],
		["Do de novo truncating mutations in WASF1 cause cancer?", "De Novo Truncating Mutations in WASF1 Cause Intellectual Disability with Seizures.", ["No"]],
		["Is the tyrosine kinase BTK implicated in autoimmunity?", "Augmented TLR9-induced Btk activation in PIR-B-deficient B-1 cells provokes excessive autoantibody production and autoimmunity.", ["Yes"]]
	];

	const factoid_examples = [
		["Which receptor is modulated with Siponimod?", "We validated several known promyelinating compounds and demonstrated that the strong remyelinating efficacy of siponimod implicates the sphingosine-1-phosphate receptor 5", ["sphingosine-1-phosphate"]],
		["Which clotting factor is in the Andexxa?", "Intravenous andexanet alfa [coagulation factor Xa (recombinant), inactivated-zhzo; Andexxa\u00ae] is a first-in-class recombinant modified factor Xa protein that has been developed by Portola Pharmaceuticals as a universal antidote to reverse anticoagulant effects of direct or indirect factor Xa inhibitors.", ["Xa"]],
		["Cushing's disease is associated with a tumor in what part of the body?", "Cushing's disease (CD) is a rare disabling condition caused by Adrenocorticotropic hormone (ACTH)-secreting adenomas of the pituitary", ["pituitary"]],
		["Which domain of the MOZ/MYST3 protein complex associates with histone H3?", "In conclusion, our data show that Moz regulates H3K9 acetylation at Hox gene loci and that RA can act independently of Moz to establish specific Hox gene expression boundaries. The double PHD finger domain of MOZ/MYST3 induces \u03b1-helical structure of the histone H3 tail to facilitate acetylation and methylation sampling and modification", ["double PHD finger domain"]],
		["Which company sells the drug Afrezza since 2015?", "In contrary, MannKind Corporation started developing its ultra-rapid-acting insulin Afrezza in a bold bid, probably by managing the issues in which Exubera was not successful. Afrezza has been marketed since February, 2015 by Sanofi after getting FDA approval in June 2014.", ["Sanofi"]],
		["Rachmilewitz Index is used for which diseases?", "At present, many endoscopic indices of ulcerative colitis have been introduced, including the Truelove and Witts Endoscopy Index, Baron Index, Powell-Tuck Index, Sutherland Index, Clinic Endoscopic Sub-Score, Rachmilewitz Index, Modified Baron Index, Endoscopic Activity Index, Ulcerative Colitis Endoscopic Index of Severity, Ulcerative Colitis Colonoscopic Index of Severity, and Modified Mayo Endoscopic Score.", ["ulcerative colitis"]],
	];

	const list_examples = [
		["List two medication included in the Juluca pill.", "Dolutegravir/rilpivirine (Juluca\u00ae) is the first two-drug single-tablet regimen (STR) to be approved for the treatment of HIV-1 infection in adults.", ["dolutegravir" , "rilpivirine"]],
		["Which drugs are included in the EE-4A regimen for Wilm\u0027s tumor?", "Five patients received treatment regimen EE-4A, dactinomycin, and vincristine.", ["dactinomycin", "vincristine"]],
		["Name two rotavirus vaccines.", "Two rotavirus vaccines, Rotateq and Rotarix, are licensed for global use; however, the protection they confer to unvaccinated individuals through indirect effects remains unknown.", ["Rotateq", "Rotarix"]],
		["Which symptoms comprise Abdominal aortic aneurysm rupture Triad?", "The correct diagnosis based on the classic triad of shock, acute abdominal pain, and pulsatile abdominal mass was made in only one of 19 (5.3%) patients. Only 50% of abdominal aortic aneurysms present with the classic triad of hypotension, back pain and a pulsatile abdominal mass. Some of these patients present with the classic triad of symptoms such as abdominal pain, pulsatile abdominal mass and shock.", ["shock", "acute abdominal pain", "pulsatile abdominal mass"]],
		["What is the Triad of Alport Syndrome?", "PURPOSE: Alport syndrome is a rare condition characterized by the clinical triad of nephritic syndrome, sensorineural deafness, and ophthalmological alterations. Alport syndrome is an oculo-renal syndrome characterized by a triad of clinical findings consisting of hemorrhagic nephritis, sensorineural hearing loss and characteristic ocular findings.", ["nephritic syndrome", "sensorineural deafness", "ophthalmological alterations"]]
	];

	function get_next_example(examples, q_type){
		let idx_of_increment = ['yesno', 'factoid', 'list'].indexOf(q_type);
		increments[idx_of_increment] = (increments[idx_of_increment] + 1) % examples.length;
		return [examples[increments[idx_of_increment]][0], examples[increments[idx_of_increment]][1], [...examples[increments[idx_of_increment]][2]]]
	}

	async function reset_example(){
		document.getElementById("first-opt").selected = true;
			question_input.val("");
			context_input.val("");
			expected_answer_input.val("");
			predicted_answer_input.val("");
	}

	function populate_example(q_type){
		reset_example().then(() => {
			let example = ["", "", ""]
			if(q_type === "yesno"){
				example = get_next_example(yesno_examples, q_type);
			} else if(q_type === "factoid"){
				example = get_next_example(factoid_examples, q_type);
			} else if(q_type === "list") { // list
				example = get_next_example(list_examples, q_type);
			} else {
				return;
			}

			console.log('Example', example);
			console.log('answer', example[2]);
			console.log(factoid_examples);
			question_type_input.val(q_type);
			const question = example[0];
			question_input.val(question);

			const context = example[1];
			context_input.val(context);

			const answer = example[2];
			let combined_answer = "";
			if(answer.length > 1){
				for(let i=0; i < answer.length; i++) {
					let cap_a = answer[i].charAt(0).toUpperCase() + answer[i].slice(1);
					combined_answer += (i + 1).toString() + ". " + cap_a + "\n";
				}
			} else {
				combined_answer = answer.pop();
			}
			expected_answer_input.val(combined_answer.charAt(0).toUpperCase() + combined_answer.slice(1));

			});

	}

	//buttons
	const reset_button = document.getElementById("reset-button");
	let factoid_button = document.getElementById("factoid-button");
	let yesno_button = document.getElementById("yesno-button");
	let list_button = document.getElementById("list-button");

	reset_button.addEventListener('click', function (){ return populate_example("reset") });
	factoid_button.addEventListener('click', function (){ return populate_example("factoid")});
	yesno_button.addEventListener('click', function (){ return populate_example("yesno")});
	list_button.addEventListener('click', function (){ return populate_example("list")});


	function invalid_input(element){
		element.css("box-shadow", "0 0 10px rgb(255, 0, 0)");
	}

	// When the user ID form is submitted, POST input to /process and if the user ID exists, redirect to welcome page
	$('#form').on('submit', function(event) {
		$('#answer-input').val("");
		$.ajax({
			data: {
			question: question_input.val(),
			context: context_input.val(),
			question_type: question_type_input.val()
			},
			type: 'POST',
			url: '/handlepost',
			success: function (response) {
				if (!response) {
					invalid_input(question_input);
					invalid_input(context_input);
				} else {
					console.log(response)
					console.log(response["prediction"])
					let p_answer = response["prediction"];

					let c_answer = "";
					if(p_answer.length > 1){
						for(let i=0; i < p_answer.length; i++) {
							let cap_a = p_answer[i].charAt(0).toUpperCase() + p_answer[i].slice(1);
							c_answer += (i + 1).toString() + ". " + cap_a + "\n";
						}
						} else {
							c_answer = p_answer.pop();
						}
					$('#answer-input').val(c_answer.charAt(0).toUpperCase() + c_answer.slice(1));
				}
				return response;
			}
		});
		// HTML automatically tries to post the form, we therefore manually stop this
		event.preventDefault();
	});

});